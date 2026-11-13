"""Save the completed prediction-blind manual input review and final bindings."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from collect_holdout import HERE,DATA,load,save,sha


def main():
    manifest=load(DATA/'final_manifest.json')
    rows=load(DATA/'final_test.json')
    audit=load(DATA/'isolation_audit.json')
    assert len(rows)==manifest['images']==95
    assert manifest['test_sha256']==sha(DATA/'final_test.json')
    assert audit['test_sha256']==sha(DATA/'final_test.json')
    assert audit['manifest_sha256']==sha(DATA/'final_manifest.json')
    assert audit['image_SHA256_all_verified'] and not audit['overlaps']
    assert len({row['user_id'] for row in rows})==len(rows)
    assert manifest.get('any_model_inference_performed') is False
    assert sha(DATA/'test.json')==manifest['initial_test_sha256']
    assert sha(DATA/'manifest.json')==manifest['initial_manifest_sha256']
    # Root has not prepared any test features, tokens or scores. Check the known
    # artifact locations in addition to the task's explicit no-inference contract.
    forbidden=[HERE/'features/test.json',HERE/'tokens/test',HERE/'bio_tokens/test',HERE/'test_results.json']
    assert not any(path.exists() for path in forbidden)
    by_id={row['observation_id']:row for row in rows}
    def keys(ids):
        return [by_id[str(value)]['key'] for value in ids]
    pure=keys([163771529])
    full_size=[169875981,151004488,153915767,156381536,163771529,
               160690829,148589197,187880920,159068431,147300053,149706534,166385485,
               363579730,156742821]
    notes={
        '169875981': 'Bird visible in a lined plastic container, eyes closed/atypical posture. Handled/rescue context; injury/death cannot be established from this still image.',
        '151004488': 'Bird lying on the ground, apparently unresponsive. Flag suspected dead/injured; no veterinary determination from the photo.',
        '153915767': 'Bird crouched on a roadside yellow marking. Health/age uncertain; do not infer injury or fledgling status.',
        '156381536': 'Bird curled inside a cardboard box. Handled/rescue context, health uncertain.',
        '163771529': 'Clearly decomposed remains with exposed feathers/bones and no intact recognizable bird body. Source species label and main denominator retained.',
        '159068431': 'Very small/occluded dark subject near the tree base. Bird visibility cannot be established confidently by this review; keep source label and flag uncertainty rather than declaring an empty image.',
        '363579730': 'Large dark waterbird and several grey downy young-looking waterbirds in a managed pond. Do not label as mixed species solely from plumage/size differences.',
        '156742821': 'Multiple male/female-looking ducks. Individual species not relabeled by this review.',
        '186777796': 'Multiple gulls at different distances; possible mixed species cannot be expert-confirmed by this input review.',
        '180332623': 'Two white wading birds; individual species not expert-confirmed.',
        '174842717': 'Flock perched on a building; individual source-species correspondence is not guaranteed.',
        '146587637': 'Unusual wall/vertical-ground view of a dove-like bird. Pose alone is not evidence of death.',
        '145951113': 'Black letterbox bars present; original pixels retained.',
        '150545910': 'Human hand and a nearby flying gull both visible; original pixels retained.',
    }
    multiple=[153790921,173561597,156742821,180332623,174842717,186777796,179830926,
              155090015,151467299,164822858,166385485,163084323,363579730,184663937,263006277]
    sheets=[]
    for sheet in audit['contact_sheets']:
        assert sha(sheet['path'])==sheet['sha256']
        sheets.append({**sheet,'all_cells_manually_viewed':True})
    qa={'status':'passed_with_notes','inference_can_start':True,'new_inference_can_start':True,
        'review_utc':datetime.now(timezone.utc).isoformat(),'reviewed_before_inference':True,
        'any_model_inference_performed':False,'model_outputs_used':False,'training_or_selection_used':False,
        'training_or_hyperparameter_selection_used':False,'source_labels_changed':False,'denominator_changed':False,
        'manifest_sha256':sha(DATA/'final_manifest.json'),'test_sha256':sha(DATA/'final_test.json'),
        'isolation_audit_sha256':sha(DATA/'isolation_audit.json'),'images':len(rows),
        'species':len({row['scientific_name'] for row in rows}),'photographers':len({row['user_id'] for row in rows}),
        'photographer_identity_basis':'Observation user_id proxy; not independent verification of physical photographer identity',
        'main_denominator':len(rows),'excluded_from_main':[],'contact_sheets_reviewed':sheets,
        'input_images_manually_reviewed':len(rows),'full_size_originals_reviewed':[
            {'key':by_id[str(value)]['key'],'path':by_id[str(value)]['path'],'sha256':by_id[str(value)]['sha256']} for value in full_size],
        'pure_remains_or_no_recognizable_body_keys':pure,
        'dead_or_injured_keys':pure,
        'dead_or_injured_scope':'Only clear remains are definite. Suspected health/posture cases are separately marked, not clinically diagnosed.',
        'suspected_dead_or_injured_keys':keys([151004488]),
        'boxed_or_handled_keys':keys([169875981,156381536]),
        'health_or_age_uncertain_keys':keys([153915767]),
        'bird_visibility_uncertain_keys':keys([159068431]),
        'multi_species_keys':[],
        'multi_species_scope':'No mixed-species assertion was expert-confirmed; multiple individuals are not automatically multiple species.',
        'possible_mixed_species_or_individual_label_ambiguity_keys':keys([186777796,180332623,174842717]),
        'multiple_individual_keys':keys(multiple),
        'predefined_secondary_excluding_pure_remains':{'denominator':len(rows)-len(pure),'excluded_keys':pure,
            'interpretation':'Excludes only the pre-declared definite remains; does not certify all remaining images as pristine live birds.'},
        'notes':[{'key':by_id[value]['key'],'note':note} for value,note in notes.items()],
        'expert_species_reidentification_performed':False,
        'source_label_basis':'Public research-grade observation species; not per-individual expert labels',
        'quality_flags_selected_before_any_prediction':True,
        'class_coverage_is_37_not_80':True,'desired_120_to_160_target_met':False,
        'collection_stopped_after_authorized_extra_window':True,
        'near_duplicate_or_capture_sequence_isolation_guaranteed':False,
        'foundation_pretraining_overlap_known':False,
        'known_test_artifacts_checked_absent':[str(path) for path in forbidden]}
    save(DATA/'input_qa.json',qa)
    initial=load(DATA/'manifest.json')
    errors=initial['query_errors']+manifest['supplement_query_errors']
    credits=Counter(row['attribution'] for row in rows)
    save(DATA/'completion.json',{'status':'frozen_and_prediction_blind_input_QA_completed',
        'final_test_sha256':sha(DATA/'final_test.json'),'final_manifest_sha256':sha(DATA/'final_manifest.json'),
        'input_qa_sha256':sha(DATA/'input_qa.json'),'images':len(rows),'species':qa['species'],
        'observer_users':qa['photographers'],'all_test':True,'any_model_inference_performed':False,
        'API_queries':manifest['total_API_queries'],'API_query_errors':len(errors),
        'photo_download_or_duplicate_rejections':len(initial['rejected'])+len(manifest['rejected']),
        'licenses':dict(Counter(row['license'] for row in rows)),
        'duplicate_exact_attribution_strings':{text:n for text,n in credits.items() if n>1},
        'source_labels_and_main_denominator_unchanged':True,
        'desired_120_to_160_image_target_met':False})
    lines=['# 新独立留出采集与输入QA','',
        '**最终95张、37种、95个不同观察作者，全部仅作test，尚未做任何模型推理或评分。**未达到希望的120–160张；没有通过重复作者或扩大类别凑数。',
        '',f"共执行{manifest['total_API_queries']}个预定/获准补充API查询，{len(errors)}个查询出现网络超时；初批24张下载失败或精确重复被记录。额外2024–2025窗口未产生合格新候选，按约定结束采集。",
        '', '最终入口为final_test.json和final_manifest.json；初批test.json、manifest.json及摘要保留。source_files绑定历史32+已测128聚合，并显式重查1577训练及388验证清单。user/observation/photo/SHA/exact dHash与这些源无重叠，测试内部五项也唯一，每种最多4张。作者独立使用观察user_id作为代理，署名原样保留。',
        '', '4张接触表的95个格均已人工查看，14张原图额外放大。许可和署名、观察链接、原图SHA已全部核验。源照片标签是观察级研究级标签，未专家重判逐只鸟种。',
        '', '## 预测前固定的质量标记','',
        '- 纯残骸／无完整可辨鸟体：FT_HOLDOUT_163771529，保留在95张主分母。',
        '- 卧倒疑似死伤：FT_HOLDOUT_151004488，仅作疑似标记。',
        '- 容器／盒中个体：FT_HOLDOUT_169875981、FT_HOLDOUT_156381536；不从静态姿态诊断生死。',
        '- 健康或年龄不确定：FT_HOLDOUT_153915767。',
        '- 鸟体可见性难以人工确定：FT_HOLDOUT_159068431；源标签保留，不直接当作空图。',
        '- 多个体与可能混合物种分别记录，未专家确认的情况只在notes说明。',
        '', '如需要次级统计，可固定排除已在预测前标注的1张纯残骸，形成94张附加口径；这不意味着其余94张全是清楚活体。主指标始终使用95张，不据预测改分母、标签或质量名单。',
        '', 'input_qa.json已绑定final_manifest、final_test及隔离核验摘要，并开放inference_can_start。任何训练或选参不得使用这些图；模型选择仍只用原训练/验证。基础预训练重叠及穷尽的近重复/拍摄序列隔离无法由本次元数据核验保证。']
    (DATA/'采集与输入QA说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('FINAL INPUT QA PASSED WITH NOTES:',len(rows),'images;',qa['species'],'species; inference_can_start=true',flush=True)


if __name__=='__main__':
    main()
