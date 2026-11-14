"""Score-independent routing for separate, advisory species reviewers.

``top_ids`` is the original classifier's ordered candidate list. Only its first
five entries participate. A group is eligible when Top1 belongs to it, or when
Top1 is a registered outgroup neighbor AND Top5 contains a group member.

``confusion_neighbors`` is an optional preconfigured mapping of group ID to
outgroup species IDs, for example ``{'willow4': [50]}``. Its default is empty;
this module never learns neighbors from observed test errors. Group matching
does not itself authorize advice: the frozen availability state is checked
after conflict detection. No confidence, area, model or image is consumed.
"""
from collections.abc import Mapping, Sequence
from numbers import Integral


VOCABULARY_SIZE = 1343
GROUPS = [
    {
        'group_id': 'willow4',
        'ids': [70, 126, 198, 473],
        'state': 'experimental_advisory',
        'advice_available': True,
        'label': '柳莺组（冻结实验建议）',
    },
    {
        'group_id': 'thrush_complex',
        'ids': [84, 224, 90, 246, 118, 537],
        'state': 'no_advice',
        'advice_available': False,
        'label': '鸫类组（冻结为不提供建议）',
    },
    {
        'group_id': 'sparrow_pair',
        'ids': [1, 88],
        'state': 'no_advice',
        'advice_available': False,
        'label': '麻雀组（冻结为不提供建议）',
    },
]

STATUS_LABELS = {
    'disabled': '复核已关闭',
    'routed': '进入独立实验复核',
    'not_triggered': '未触发相似鸟复核',
    'not_enabled': '该组未启用建议',
    'conflict': '多个复核组冲突',
    'invalid_input': '候选或路由配置无效',
}

_MESSAGES = {
    'global_switch_off': '相似鸟复核已关闭，保留原识别结果。',
    'invalid_enabled': '复核开关须为布尔值；本次不运行复核，保留原识别。',
    'invalid_top_ids': '原候选须为有序、互不重复的有效鸟种 ID；本次不运行复核。',
    'invalid_group_config': '复核组配置无效；本次不运行复核，保留原识别。',
    'invalid_confusion_neighbors': '预先注册的混淆邻接表无效；本次不运行复核。',
    'empty_candidates': '原识别没有鸟种候选，本次不运行复核。',
    'no_matching_group': '原 Top1 未命中复核组或其预先注册的混淆邻接关系。',
    'multiple_groups_matched': '候选同时命中多个复核组；为避免混合结果，本次均不运行。',
    'frozen_no_advice': '该组验证未显示可安全采用的纠正收益，冻结为不提供建议；保留原判。',
    'advice_unavailable': '该组暂无可用的冻结建议，保留原判。',
    'top1_group_member': '原 Top1 属于该组，进入独立实验复核；建议不覆盖原识别。',
    'registered_neighbor_with_group_top5':
        '原 Top1 命中预先注册的组外邻接，且 Top5 含组成员；进入独立实验复核。',
}


def _result(status, reason, group=None, matched=()):
    return {
        'status': status,
        'status_label': STATUS_LABELS[status],
        'reason': reason,
        'message': _MESSAGES[reason],
        'group_id': group['group_id'] if group else None,
        'group_label': group['label'] if group else None,
        'matched_group_ids': list(matched),
        'advice_available': bool(group and group['advice_available']
                                 and group['state'] != 'no_advice'),
    }


def _ids(value):
    """Copy and validate ordered IDs; booleans/strings are not species IDs."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError('An ordered ID sequence is required')
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise ValueError('Species ID is not an integer')
        number = int(item)
        if not 0 <= number < VOCABULARY_SIZE:
            raise ValueError('Species ID outside the frozen global vocabulary')
        result.append(number)
    if len(result) != len(set(result)):
        raise ValueError('Duplicate species IDs')
    return result


def _groups(groups):
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes, bytearray)):
        raise ValueError('A group sequence is required')
    result, names = [], set()
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError('Invalid group record')
        name, label = group.get('group_id'), group.get('label')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Missing/repeated group ID')
        if not isinstance(label, str) or not label:
            raise ValueError('Missing group label')
        members = _ids(group.get('ids'))
        if not members or group.get('state') not in ('experimental_advisory', 'no_advice'):
            raise ValueError('Missing group members/state')
        if type(group.get('advice_available')) is not bool:
            raise ValueError('Availability must be explicit')
        # Use our own records so normalization never mutates caller data.
        result.append({'group_id': name, 'label': label, 'ids': members,
                       'state': group['state'], 'advice_available': group['advice_available']})
        names.add(name)
    return result


def _neighbors(value, groups):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError('A group-to-outgroup-neighbor mapping is required')
    members = {group['group_id']: set(group['ids']) for group in groups}
    result = {}
    for name, species in value.items():
        if name not in members:
            raise ValueError('Unknown neighbor group')
        neighbors = set(_ids(species))
        if neighbors & members[name]:
            raise ValueError('Registered neighbor must be outside its own group')
        result[name] = neighbors
    return result


def route_candidate(top_ids, *, enabled, groups=GROUPS, confusion_neighbors=None):
    """Return a safe routing decision without changing inputs or predictions.

    Only ``status == 'routed'`` authorizes invoking a reviewer. ``not_enabled``
    still names the matched frozen group. ``conflict`` names all matched group
    IDs but deliberately selects none. Invalid inputs return ``invalid_input``
    instead of running a model or raising an ordinary validation error.
    """
    if enabled is False:
        return _result('disabled', 'global_switch_off')
    if type(enabled) is not bool:
        return _result('invalid_input', 'invalid_enabled')
    try:
        candidates = _ids(top_ids)
    except (TypeError, ValueError, OverflowError):
        return _result('invalid_input', 'invalid_top_ids')
    if not candidates:
        return _result('not_triggered', 'empty_candidates')
    try:
        configured = _groups(groups)
    except (TypeError, ValueError, OverflowError):
        return _result('invalid_input', 'invalid_group_config')
    try:
        neighbors = _neighbors(confusion_neighbors, configured)
    except (TypeError, ValueError, OverflowError):
        return _result('invalid_input', 'invalid_confusion_neighbors')
    top1, top5 = candidates[0], set(candidates[:5])
    matches = []
    for group in configured:
        members = set(group['ids'])
        if top1 in members:
            matches.append((group, 'top1_group_member'))
        elif top1 in neighbors.get(group['group_id'], ()) and members & top5:
            matches.append((group, 'registered_neighbor_with_group_top5'))
    matched_ids = [group['group_id'] for group, _ in matches]
    if len(matches) > 1:
        return _result('conflict', 'multiple_groups_matched', matched=matched_ids)
    if not matches:
        return _result('not_triggered', 'no_matching_group')
    group, reason = matches[0]
    if group['state'] == 'no_advice':
        return _result('not_enabled', 'frozen_no_advice', group, matched_ids)
    if not group['advice_available']:
        return _result('not_enabled', 'advice_unavailable', group, matched_ids)
    return _result('routed', reason, group, matched_ids)


def describe_row(row):
    """Short Chinese status suitable for a separate GUI review column."""
    if not isinstance(row, Mapping):
        return STATUS_LABELS['invalid_input']
    label = row.get('group_label')
    message = row.get('message') or _MESSAGES.get(row.get('reason'))
    if not message:
        message = STATUS_LABELS.get(row.get('status'), '未知复核状态，保留原识别。')
    return f'{label}：{message}' if label else message
