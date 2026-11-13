"""Read-only CPU verification of the two completed GPU CLI demonstrations."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageOps
from inference.engine_n import square_context

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PACKAGE=Path(r'D:\项目\鸟类识别\识别模型\v1\v1.1\NVIDIA')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    selection_path=HERE/'selection.json'
    selection=read(selection_path)
    selection_sha=sha(selection_path)
    old_path=ROOT/'detection_accuracy_20261113/cli_smoke/result.json'
    old=read(old_path)
    old_primary=[i for i,b in enumerate(old['birds']) if b['primary']]
    assert len(old_primary)==1
    labels=read(PACKAGE/'configs/china_labels.json')
    validation_path=ROOT/'dinov2_species_20261113/features/n_val.json'
    validation=read(validation_path)
    records=[]
    for name in ('cli_smoke_dino','cli_smoke_bio'):
        path=HERE/name/'result.json'
        assert path.is_file(),'Waiting for both completed CLI exports: '+str(path)
        result=read(path)
        assert result['selection_sha256']==selection_sha
        spec=next(item for item in selection['candidates'] if item['id']==result['candidate_id'])
        assert result['candidate_spec']==spec
        assert Path(result['source_image']).resolve()==Path(old['source_image']).resolve()
        assert result['source_sha256']==old['source_sha256']==sha(result['source_image'])
        assert result['detection_mode']=='normal' and result['YOLO_detection_executed'] is True
        assert result['candidate_count']==len(result['candidates'])==len(old['birds'])
        assert result['primary_index']==old_primary[0]
        assert [b['xyxy'] for b in result['candidates']]==[b['xyxy'] for b in old['birds']]
        assert [b['accepted_original_E'] for b in result['candidates']]==[b['confirmed'] for b in old['birds']]
        assert [b['primary'] for b in result['candidates']]==[b['primary'] for b in old['birds']]
        assert result['count_is_unique_individual_count'] is False
        source_row=[row for row in validation if Path(row['path']).resolve()==Path(result['source_image']).resolve()]
        assert len(source_row)==1
        source_row=source_row[0]
        primary=result['candidates'][result['primary_index']]
        assert source_row['bbox']==primary['xyxy']
        assert source_row['accepted']==primary['accepted_original_E']
        with Image.open(result['source_image']) as original:
            image=ImageOps.exif_transpose(original).convert('RGB')
        crops=[]
        for number,(candidate,reference) in enumerate(zip(result['candidates'],old['birds'])):
            assert candidate['candidate_index']==number
            assert candidate['original_detection']['xyxy']==reference['xyxy']
            assert candidate['original_species_info']['accepted']==reference['confirmed']
            assert candidate['confirmed']==reference['confirmed']
            if not candidate['confirmed']:
                assert candidate['display_name']=='鸟（种类不确定）'
            expected=square_context(image,candidate['xyxy'])
            crop_path=HERE/name/candidate['crop']
            assert sha(crop_path)==candidate['crop_sha256']
            with Image.open(crop_path) as actual:
                assert actual.format=='PNG' and actual.mode=='RGB'
                assert actual.size==expected.size and actual.tobytes()==expected.tobytes()
                crops.append({'index':number,'path':str(crop_path),'sha256':sha(crop_path),
                              'size':list(actual.size),'RGB_pixels_identical_to_square_context':True,
                              'scientific_name':candidate['scientific_name'],
                              'accepted_original_E':candidate['accepted_original_E'],'primary':candidate['primary']})
        with Image.open(HERE/name/'annotated.jpg') as rendered:
            assert rendered.size==image.size
            rendered.verify()
        scored_path=HERE/'runs/validation/n_val'/(spec['id']+'.json')
        scored=read(scored_path)
        matches=[row for row in scored['records'] if row['key']==source_row['key']]
        assert len(matches)==1
        measured=matches[0]
        assert measured['has_box'] and measured['prediction']==primary['species_global_id']
        assert labels[measured['prediction']]['scientific_name']==primary['scientific_name']
        assert measured['accepted']==primary['accepted_original_E']
        probability_path=Path(scored['probabilities']['path'])
        assert sha(probability_path)==scored['probabilities']['sha256']
        with np.load(probability_path,allow_pickle=False) as data:
            key_matches=np.flatnonzero(data['keys']==source_row['key'])
            assert len(key_matches)==1
            index=int(key_matches[0])
            assert data['present'][index]
            assert int(data['predictions'][index])==primary['species_global_id']
            values=data['probabilities'][index]
            classes=data['class_ids']
            assert classes.tolist()==result['classification_global_ids']
            difference=[]
            for item in primary['top5']:
                positions=np.flatnonzero(classes==item['global_id'])
                assert len(positions)==1
                value=float(values[int(positions[0])])
                difference.append({'global_id':item['global_id'],'scientific_name':item['scientific_name'],
                    'CLI_probability':item['probability'],'validation_probability':value,
                    'absolute_difference':abs(item['probability']-value)})
        model_checks=[]
        for model in spec['models']:
            digest=sha(model['checkpoint'])
            assert digest==model['checkpoint_sha256']
            model_checks.append({'seed':model['seed'],'checkpoint':model['checkpoint'],'sha256':digest})
        records.append({'folder':name,'result_path':str(path),'result_sha256':sha(path),
            'candidate_id':spec['id'],'architecture':spec['architecture'],'source_key':source_row['key'],
            'source_sha256':result['source_sha256'],'candidate_count':len(crops),'primary_index':result['primary_index'],
            'all_original_box_coordinates_identical':True,'primary_and_original_E_gates_identical':True,
            'crops':crops,'annotated_sha256':sha(HERE/name/'annotated.jpg'),
            'validation_record_sha256':sha(scored_path),'validation_probabilities_sha256':sha(probability_path),
            'primary_prediction_matches_frozen_validation':True,'primary_name':primary['scientific_name'],
            'primary_top3_matches_validation':measured['top3']==[r['global_id'] for r in primary['top5'][:3]],
            'primary_top5_probability_comparison':difference,
            'primary_exported_top5_max_absolute_difference':max(row['absolute_difference'] for row in difference),
            'full_probability_vector_comparison_available':False,
            'nonprimary_validation_reference_available':False,
            'probability_note':'CLI exports Top-5 only. Recorded differences may include BF16 batch-dependent math; no exact-bit equality assertion.',
            'checkpoint_files_verified':model_checks,'reported_timings_ms':result['timings_ms']})
    code=[HERE/name for name in ('predict.py','network.py','bio_tail.py','scoring.py','common.py')]
    code += [ROOT/'dinov2_species_20261113'/name for name in ('evaluate.py','encoders.py')]
    code += [PACKAGE/'app'/name for name in ('engine.py','inference/engine_n.py','inference/encoders.py')]
    report={'status':'passed','scope':'Two completed GPU demonstrations on the same old validation photo; this review is CPU-only',
            'old_validation_source_key':records[0]['source_key'],'fresh95_images_read_or_scored':False,
            'model_inference_performed_by_review':False,'torch_imported_by_review':'torch' in sys.modules,
            'selection_sha256':selection_sha,'old_CLI_reference_sha256':sha(old_path),
            'validation_rows_sha256':sha(validation_path),'code_sha256':{str(path):sha(path) for path in code},
            'review_source_sha256':sha(__file__),'records':records,
            'candidate_count_is_not_unique_bird_count':True,'production_entry_replaced':False,
            'performance_improvement_claimed':False,
            'note':'A successful one-photo runtime check validates integration and crop/gate preservation, not accuracy improvement or batch-exact probability parity.'}
    destination=HERE/'data_review/cli_runtime_review.json'
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 两个实验入口运行产物核查','',
           'DINO与Bio尾块入口均已在旧验证原图N_263246209完成实际GPU运行；本核查只读文件、CPU比较，没有重复推理，也未读取fresh95。',
           '', '| 候选 | 原框／primary／E gate | PNG与square_context像素 | 主预测与冻结验证 | 已导出Top-5最大概率差 |',
           '|---|---|---|---|---:|']
    for row in records:
        lines.append(f"| {row['candidate_id']} | 两框／0／true,false完全一致 | 逐像素相同 | {row['primary_name']}，一致 | {row['primary_exported_top5_max_absolute_difference']:.10g} |")
    lines += ['', '原图其实只有一只牛背鹭而有两个保留候选框，数量仍称候选数。第二框保持不确认，显示“鸟（种类不确定）”。',
              '', '验证缓存只保存主框分类结果，因此只对主框比较冻结候选预测与概率；没有宣称次框也有验证参考。CLI只导出Top-5，未声称完整概率向量逐位一致，概率差仅记录为数值差异。',
              '', 'JSON绑定两个result、selection、旧CLI、验证JSON/NPZ、相关源码与六个尾块检查点摘要。本次单图可运行性通过不等于准确率提高，正式入口保持原样。']
    (HERE/'data_review/CLI运行核查.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    assert sha(selection_path)==selection_sha
    print('CLI_RUNTIME_REVIEW_PASSED',[(r['candidate_id'],r['primary_exported_top5_max_absolute_difference']) for r in records],flush=True)


if __name__=='__main__':
    main()
