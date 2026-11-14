"""Standard-library boundaries for score-independent advisory routing.

Run from the repository root:
    python -B -m unittest desktop_review.test_routing -v
"""
from copy import deepcopy
import inspect
import unittest

try:
    from .routing import GROUPS, STATUS_LABELS, describe_row, route_candidate
except ImportError:
    from routing import GROUPS, STATUS_LABELS, describe_row, route_candidate


class RoutingTests(unittest.TestCase):
    def route(self, top_ids, **kwargs):
        result = route_candidate(top_ids, enabled=True, **kwargs)
        self.assertTrue({'status', 'reason', 'group_id', 'group_label',
                         'matched_group_ids'} <= result.keys())
        return result

    def test_disabled_has_priority_and_never_routes_even_bad_candidates(self):
        for candidates in ([70, 126], [84, 224], None, 'bad'):
            with self.subTest(candidates=candidates):
                result = route_candidate(candidates, enabled=False)
                self.assertEqual(result['status'], 'disabled')
                self.assertIsNone(result['group_id'])
                self.assertEqual(result['matched_group_ids'], [])

    def test_all_frozen_willow_members_route(self):
        for species in [70, 126, 198, 473]:
            with self.subTest(species=species):
                result = self.route([species, 50, 59, 72, 0])
                self.assertEqual(result['status'], 'routed')
                self.assertEqual(result['reason'], 'top1_group_member')
                self.assertEqual(result['group_id'], 'willow4')

    def test_frozen_no_advice_groups_never_authorize_inference(self):
        for group_id, species in [('thrush_complex', 84), ('sparrow_pair', 88)]:
            with self.subTest(group=group_id):
                result = self.route([species, 70, 126])
                self.assertEqual(result['status'], 'not_enabled')
                self.assertEqual(result['reason'], 'frozen_no_advice')
                self.assertEqual(result['group_id'], group_id)
                self.assertEqual(result['matched_group_ids'], [group_id])
                self.assertFalse(result['advice_available'])

    def test_high_confidence_and_small_area_cannot_bypass_member_routing(self):
        # Upstream score/area metadata is deliberately absent from the API.
        decisions = [self.route(row['top_ids']) for row in (
            {'top_ids': [70, 50, 59, 72, 0], 'confidence': .999999, 'area': .000001},
            {'top_ids': [70, 50, 59, 72, 0], 'confidence': .12, 'area': .8})]
        self.assertEqual(decisions[0], decisions[1])
        self.assertEqual(decisions[0]['status'], 'routed')
        parameters = inspect.signature(route_candidate).parameters
        self.assertNotIn('confidence', parameters)
        self.assertNotIn('area', parameters)

    def test_outgroup_top5_members_alone_do_not_trigger(self):
        result = self.route([50, 70, 84, 88, 126])
        self.assertEqual(result['status'], 'not_triggered')
        self.assertEqual(result['matched_group_ids'], [])

    def test_registered_outgroup_neighbor_and_top5_member_trigger(self):
        result = self.route([50, 70, 59, 72, 0], confusion_neighbors={'willow4': [50]})
        self.assertEqual(result['status'], 'routed')
        self.assertEqual(result['reason'], 'registered_neighbor_with_group_top5')
        self.assertEqual(result['group_id'], 'willow4')

    def test_registered_neighbor_needs_member_within_first_five(self):
        for top_ids in ([50, 59, 72, 0, 3], [50, 59, 72, 0, 3, 70]):
            with self.subTest(top_ids=top_ids):
                self.assertEqual(self.route(top_ids, confusion_neighbors={
                    'willow4': [50]})['status'], 'not_triggered')

    def test_registered_neighbor_cannot_enable_frozen_no_advice_group(self):
        result = self.route([50, 84, 59], confusion_neighbors={'thrush_complex': [50]})
        self.assertEqual(result['status'], 'not_enabled')
        self.assertEqual(result['group_id'], 'thrush_complex')

    def test_multiple_registered_groups_conflict_even_when_one_unavailable(self):
        result = self.route([50, 70, 84, 59], confusion_neighbors={
            'willow4': [50], 'thrush_complex': [50]})
        self.assertEqual(result['status'], 'conflict')
        self.assertIsNone(result['group_id'])
        self.assertFalse(result['advice_available'])
        self.assertEqual(result['matched_group_ids'], ['willow4', 'thrush_complex'])

    def test_direct_group_plus_other_neighbor_route_conflicts(self):
        result = self.route([70, 84, 59], confusion_neighbors={'thrush_complex': [70]})
        self.assertEqual(result['status'], 'conflict')
        self.assertEqual(result['matched_group_ids'], ['willow4', 'thrush_complex'])

    def test_empty_candidates_are_not_a_model_trigger(self):
        for top_ids in ([], ()):
            self.assertEqual(self.route(top_ids)['status'], 'not_triggered')
            self.assertEqual(self.route(top_ids)['reason'], 'empty_candidates')

    def test_bad_global_ids_fail_safely_without_coercion(self):
        for top_ids in (None, '70', {70}, {0: 70}, [True], [-1], [1343],
                        [70.0], ['70'], [70, 70], [70, None]):
            with self.subTest(top_ids=top_ids):
                result = self.route(top_ids)
                self.assertEqual(result['status'], 'invalid_input')
                self.assertEqual(result['reason'], 'invalid_top_ids')
                self.assertIsNone(result['group_id'])

    def test_invalid_switch_does_not_implicitly_enable(self):
        for enabled in ('false', 1, 0, None):
            result = route_candidate([70], enabled=enabled)
            self.assertEqual(result['status'], 'invalid_input')
            self.assertEqual(result['reason'], 'invalid_enabled')

    def test_bad_neighbor_configuration_fails_safely(self):
        for neighbors in ([50], {'unknown': [50]}, {'willow4': [70]},
                          {'willow4': [-1]}, {'willow4': ['50']}):
            with self.subTest(neighbors=neighbors):
                result = self.route([50, 70], confusion_neighbors=neighbors)
                self.assertEqual(result['status'], 'invalid_input')
                self.assertEqual(result['reason'], 'invalid_confusion_neighbors')

    def test_overlapping_group_members_conflict_instead_of_first_match_wins(self):
        groups = deepcopy(GROUPS)
        groups[1]['ids'].append(70)
        result = self.route([70], groups=groups)
        self.assertEqual(result['status'], 'conflict')
        self.assertIsNone(result['group_id'])

    def test_bad_group_configuration_is_not_a_trigger(self):
        for field, value in (('group_id', 'willow4'), ('state', 'unknown')):
            groups = deepcopy(GROUPS)
            groups[1][field] = value
            self.assertEqual(self.route([70], groups=groups)['reason'], 'invalid_group_config')

    def test_inputs_and_group_defaults_are_unchanged(self):
        top_ids = [50, 70, 84, 88, 59]
        groups = deepcopy(GROUPS)
        neighbors = {'willow4': [50], 'thrush_complex': [50]}
        original = deepcopy((top_ids, groups, neighbors, GROUPS))
        self.route(top_ids, groups=groups, confusion_neighbors=neighbors)
        self.assertEqual((top_ids, groups, neighbors, GROUPS), original)

    def test_chinese_status_description_names_group_and_block_reason(self):
        result = self.route([1, 88])
        description = describe_row(result)
        self.assertIn('麻雀组', description)
        self.assertIn('验证未显示', description)
        self.assertEqual(STATUS_LABELS['not_enabled'], '该组未启用建议')
        self.assertEqual(describe_row(None), STATUS_LABELS['invalid_input'])


if __name__ == '__main__':
    unittest.main()
