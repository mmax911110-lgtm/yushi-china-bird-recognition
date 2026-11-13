"""Record complete post-score visual review without relabeling or pipeline edits."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
EXP=HERE.parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


# Each note was assigned after viewing the full original + exact fixed1.2 crop card.
notes={
 1:('no_crop_visible_bird',['water_reflection','low_detail'],
     '原图中水面鸟体可辨，黑白强反光/倒影纹理干扰明显；检测零框，没有分类裁图。'),
 2:('no_crop_visible_bird',['boxed_or_handled','health_status_uncertain','atypical_pose'],
     '箱内完整鸟体、头喙和尾羽可见，姿势非常规；静态图不能确认生死或伤情。检测零框。'),
 3:('low_information_species_error',['flying_silhouette','small_source_crop','fine_grained_uncertain'],
     '主框确实覆盖空中鸟体，固定裁图仅50×50像素，主要为背光飞行剪影。源标苍鹭、模型夜鹭的细粒度区别不能在本图上由非专家可靠重判。'),
 4:('correct_report_but_other_models_change',['rear_view','fine_grained_uncertain'],
     '鸟体和长尾在正确主框内，背向镜头，喙部等特征未充分呈现。原部署/预选full按源标签正确；所有新闭集组将红嘴蓝鹊误改为灰喜鹊。'),
 5:('no_crop_partial_bird_shape',['vegetation_occlusion','low_contrast','source_subject_visibility_limited'],
     '查看原图后，树根左侧草丛/阴影中有局部低对比鸟形，部分被树干遮挡；无法由此重新确认物种。不是把水中花梗当作目标。检测零框。'),
 6:('wrong_individual_primary',['multi_species_scene','tiny_source_crop','source_subject_outside_primary'],
     '前景有明显的源标鸳鸯个体，但实际主框选中了远处飞行小鸟，固定裁图仅25×25像素。不能将该裁图的物种输出直接当成前景鸳鸯的分类能力。accepted=false。'),
 7:('no_crop_visible_bird',['motion_blur','shadow','partial_scene_obstruction'],
     '长椅下黑白长尾鸟体明显，存在运动模糊和阴影；检测零框，无分类裁图。'),
 8:('no_crop_visible_bird',['vegetation_occlusion','atypical_pose','cluttered_background'],
     '花叶间鸟体可见，姿势向下/侧倒，枝叶密集；检测零框，无分类裁图。'),
 9:('correct_report_but_other_models_change',['rear_side_view','fine_grained_uncertain','source_identity_not_expert_verified'],
     '裁图内有翠鸟外形、长喙及蓝色背翼，但背侧视角和低清影响特征；不重新鉴定具体种或源标签。原部署/预选full按源标签正确，仅E0全词表输出不同。'),
 10:('nonbird_primary',['multiple_visible_birds','leaf_false_positive','source_subject_outside_primary'],
      '原图有多只麻雀样鸟体；实际最高优先主框和73×73固定裁图为白色叶片，没有鸟体。accepted=false已拒绝确认；原始Top1仍会记录珠颈斑鸠等名称，不能称为确认鸟种。'),
 11:('no_crop_visible_bird',['fence_scene','low_contrast','camouflage'],
      '白色围栏两柱之间后方地面可见鸟体，棕色与地面接近、阴影较暗；检测零框。'),
 12:('no_crop_visible_bird',['flying_pose','motion_blur'],
      '草地上方展开翅膀的黑白鸟体明显，飞行姿态非常规；检测零框。'),
 13:('juvenile_pose_species_error',['boxed_or_handled','possible_juvenile','health_status_uncertain','fine_grained_uncertain'],
      '396×396固定裁图完整包含箱内鸟体，灰色/未充分成形羽貌和蹲伏姿态与典型成鸟有差别。源标鹊鸲、模型白鹡鸰；不凭此重标，也不确认健康或年龄。'),
 14:('low_information_species_error',['heavy_blur','small_source_crop','fine_grained_uncertain','unconfirmed_species_output'],
      '主框包含远处黑白鸟形，但固定裁图54×54且严重模糊。原部署按源标签喜鹊正确，预选full误改灰椋鸟；accepted=false，因此是未确认的原始Top1退化。'),
 15:('low_information_species_error',['backlit_or_dark','small_source_crop','fine_grained_uncertain'],
      '主框覆盖树上真实黑色鸟体，85×85裁图中喙部/冠部细节不清。源标八哥，乌鸦/大嘴乌鸦的输出变化均未纠正到源标签，不能因名称变动称为有效纠错。'),
 16:('suspected_nonbird_background_primary',['source_subject_outside_primary','blur','background_stake_like_object'],
      '前景草地有明显水鸟，主框却在后排水线的桩状物上；82×82裁图与相邻等距木桩外观相近，未能辨见鸟体。记为疑似非鸟/错误对象，不把低清背景物强行重新鉴种。accepted=false。'),
 17:('no_crop_visible_bird',['multiple_waterbirds','distance','water_reflection'],
      '水面多只小鸟体可见，距离较远且反光/低细节；检测零框。源标鸳鸯不是所有个体的独立真值。'),
 18:('no_crop_predeclared_remains',['decomposed_remains','predeclared_before_prediction'],
      '为腐败鸟尸/残骸，羽毛和骨骼可见，不是典型完整活鸟外观。该输入在预测前QA已预标纯残骸；保留95主分母，不据本轮结果剔除。'),
}
inputs=read(HERE/'manual_review_inputs.json')
summary_path=EXP/'runs/test/summary.json'
assert inputs['selection_sha256']==sha(EXP/'selection.json')
assert inputs['test_summary_sha256']==sha(summary_path)
assert inputs['report_candidate_sha256']==sha(EXP/'report_candidate.json')
source_index={row['key']:row for row in read(EXP/'features/test.json')}
results=read(summary_path)
report_id=inputs['report_candidate_summary']['id']
cases=[]
for item in inputs['cases']:
    category,flags,note=notes[item['review_id']]
    assert sha(item['path'])==item['source_sha256']
    assert sha(item['card_path'])==item['card_sha256']
    if item['fixed_crop_path']:
        assert sha(item['fixed_crop_path'])==item['fixed_crop_sha256']
        assert item['fixed_crop_geometry']['factor']==1.2
    cases.append({**item,'manually_viewed':True,'original_and_actual_fixed_crop_card_reviewed':True,
        'category':category,'visual_flags':flags,'observation':note,
        'source_label_changed':False,'expert_species_reidentification_performed':False})
reviewed={item['key'] for item in cases}
required=set(inputs['required_union_keys'])
assert reviewed==required and len(cases)==18
assert set(inputs['report_candidate_error_keys'])<=reviewed
assert set(inputs['changed_prediction_keys'])<=reviewed
error_categories=Counter(row['category'] for row in cases if row['report_candidate_error'])
actual_changes={row['key'] for row in cases if row['all_prediction_changes']}
assert actual_changes==set(inputs['changed_prediction_keys']) and len(actual_changes)==7
result={
    'status':'completed_complete_error_and_change_visual_review',
    'review_utc':datetime.now(timezone.utc).isoformat(),
    'reviewed_after_predictions':True,
    'selection_sha256':inputs['selection_sha256'],'test_summary_sha256':inputs['test_summary_sha256'],
    'report_candidate_sha256':inputs['report_candidate_sha256'],
    'report_candidate_summary':inputs['report_candidate_summary'],
    'manual_review_inputs_sha256':sha(HERE/'manual_review_inputs.json'),
    'source_rows_sha256':inputs['source_rows_sha256'],
    'input_QA_sha256':sha(EXP/'holdout_data/input_qa.json'),
    'primary_denominator':95,'source_labels_changed':False,'denominator_changed':False,
    'models_changed':False,'thresholds_changed':False,'crops_or_primary_boxes_changed':False,
    'expert_species_reidentification_performed':False,
    'required_union_count':18,'reviewed_count':18,'missing_required_keys':[],
    'required_changed_prediction_images':7,'report_candidate_errors':16,
    'all_report_candidate_errors_covered':True,'all_full_and_closed_name_changes_covered':True,
    'no_crop_images_reviewed':9,'fixed_crop_images_reviewed':9,'fixed_crop_factor':1.2,
    'additional_full_size_originals_reviewed':['FT_HOLDOUT_159068431','FT_HOLDOUT_174405097','FT_HOLDOUT_185531790','FT_HOLDOUT_153909307'],
    'report_candidate_error_category_counts':dict(error_categories),
    'preselected_candidate_result':results['metrics'][report_id],
    'old_deployed_full_result':results['metrics']['deployed_BioCLIP2_2_5_full'],
    'cases':cases,
    'interpretation_limits':[
        'This visual review diagnoses inputs after predictions. It is not a second expert species annotation pass.',
        'All source observation labels, all95 images and every frozen model/parameter remain unchanged.',
        'Some rawTop1 errors are unconfirmed outputs; accepted=false must not be presented as a confirmed species error.',
        'No-crop errors, a different bird in the primary box, and nonbird primary boxes cannot be attributed solely to the species classifier.',
        'Case5 contains only a partially visible low-contrast bird shape; case16 background object remains uncertain.',
        'The already predeclared remains example remains in the primary95; any94 secondary analysis must refer to pre-inference QA.'
    ]}
(HERE/'manual_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
lines=[
 '# 人工错例复核', '',
 '冻结测试未显示新骨干微调优于原部署。原部署全词表与闭集均80/95；预测前预选的研究报告候选 `tail_lr1e5_full_residual1` 为79/95，相对原部署0张纠正、1张误改。该候选在测试前已选定，部署候选保持None，没有根据测试改选表现持平的模型。', '',
 '按所有18个 full/closed 输出相对同词表原部署的具体预测名称变化取并集，再加入预选候选全部16张错误，共需复核18图；其中7图存在名称变化。已逐张查看18张原图/实际固定1.2裁图卡，并额外放大4张原图，全部覆盖。9图没有检测框，9图有实际固定裁图。没有重画框、重新裁切分类输入或运行任何模型。', '',
 '预选候选16个错误分解为9图无框、1图选中不同个体、1图主框明确为叶片、1图主框疑似水线桩状背景物、3图低信息细粒度难例、1图箱内灰色/疑似幼鸟姿态难例。这些描述不构成专家物种重标。', '',
 '两个明确的输入问题是：`FT_HOLDOUT_153909307` 的源标鸳鸯位于前景，主框却选到远处25×25飞鸟；`FT_HOLDOUT_155090015` 有多只鸟，但主框裁图为白色叶片。`FT_HOLDOUT_174405097` 的明显水鸟在前景，主框位于后排水线桩状物，记为疑似错误对象。其余分类难例包含50×50飞行剪影、54×54严重模糊和85×85暗色小鸟。', '',
 '唯一新增误改 `FT_HOLDOUT_155679526` 是喜鹊源标签样本，预选full改为灰椋鸟；该样本accepted=false，因此是原始Top1退化，不能写成“确认鸟种的新增误报”。名称从一个错误种变为另一错误种，也不能写成纠错成功。', '',
 '9张无框图中，纯残骸 `FT_HOLDOUT_163771529` 在预测前QA已固定标记；95主分母保持不变。其余包括水面反光、树根遮挡、长椅阴影、密集花叶、飞行姿态和箱内非常规姿势。箱内样本的健康/年龄状态无法从静态图确认，树根低对比局部鸟形也不据此重新鉴种。', '',
 '|序号|图片键 / 源标签|原图与实际裁图观察|',
 '|---|---|---|',
]
for row in cases:
    lines.append(f'|{row["review_id"]}|{row["key"]} / {row["source_name"]}|{row["observation"]}|')
lines += ['', '具体模型名称变化、原始/候选预测、accepted状态、框坐标、固定裁图几何、每张卡和裁图SHA均见 `manual_review.json`。证据图在 `cards/`，实际固定裁图在 `fixed_crops/`。', '',
          '本复核发生在冻结测试之后，仅解释本轮结果；没有改变源标签、95张分母、已冻结模型、阈值、框选择、裁图或部署候选。61/128旧测试及这95张测试均不能作为后续调参的未触碰终测。', '']
(HERE/'人工错例复核.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({key:result[key] for key in ('status','required_union_count','reviewed_count','required_changed_prediction_images','report_candidate_errors','report_candidate_error_category_counts')},ensure_ascii=True))
