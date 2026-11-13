"""Audit existing species fine-tuning inputs using JSON and CPU only.

Preserves all source splits, model weights and images. Writes review proposals only.
"""
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DINO=ROOT/'dinov2_species_20261113'
OLD=ROOT/'bioclip2_china_eval_20261028'
PHASE=OLD/'phase_p_bioclip25_20261106'
PACKAGE=Path(r'D:\项目\鸟类识别\识别模型\v1\v1.1\NVIDIA')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(name,value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def rows(value):
    if isinstance(value,list):
        return value
    return value['records']


def dhash(value):
    if value is None:
        return None
    text=str(value)
    return int(text,2 if len(text)==64 and set(text)<={'0','1'} else 16)


labels=read(PACKAGE/'configs/china_labels.json')
ids={row['scientific_name']:i for i,row in enumerate(labels)}
metadata={}
source_files=[
    OLD/'phase_c/manifest.json', OLD/'phase_i/manifest.json',
    OLD/'phase_n_accuracy_20261106/fresh_data/manifest.json',
    OLD/'phase_o_offline_accuracy_20261106/fresh_data/manifest.json',
    PHASE/'support_cache/records.json',
]
for path in source_files:
    for row in rows(read(path)):
        if row.get('key'):
            metadata.setdefault(row['key'],{}).update(row)
sources={name:DINO/f'features/{name}.json' for name in ('train','expanded','n_val','o_val','test')}
sources.update({'frozen_train155':DINO/'frozen_data/train.json',
                'frozen_test61':DINO/'frozen_data/test.json',
                'detection_test128':ROOT/'detection_accuracy_20261113/expanded_test_manifest.json',
                'support_all1645':OLD/'phase_l_mobile_20261106/dataset.json'})


def normalize(row):
    meta=metadata.get(row['key'],{})
    digest=row.get('sha256',row.get('source_sha256'))
    if meta:
        assert digest==meta.get('sha256',meta.get('source_sha256')),row['key']
    full={**meta,**row}
    full['sha256']=digest
    target=full.get('target')
    if target is None:
        target=ids.get(full.get('scientific_name'))
    if target is not None:
        if full.get('scientific_name'):
            assert labels[target]['scientific_name']==full['scientific_name']
        full['scientific_name']=labels[target]['scientific_name']
        full['target']=target
    for field in ('user_id','observation_id','photo_id'):
        if full.get(field) is not None:
            full[field]=str(full[field])
    if full.get('dhash') is not None:
        full['dhash']=f'{dhash(full["dhash"]):016x}'
    return full


groups={name:[normalize(row) for row in rows(read(path)) if name!='support_all1645' or row['split']=='support'] for name,path in sources.items()}
classes=sorted({row['target'] for row in groups['train']})
assert len(classes)==80 and len(groups['train'])==1434
usable=[row for row in groups['expanded'] if row['bbox'] is not None and row['target'] in classes]
groups['train_expanded1577']=groups['train']+usable
groups['validation388']=groups['n_val']+groups['o_val']
groups['validation_supported312']=[row for row in groups['validation388'] if row['target'] in classes]
groups['validation_unsupported76']=[row for row in groups['validation388'] if row['target'] not in classes]
assert len(groups['train_expanded1577'])==1577 and len(groups['validation388'])==388
assert len(groups['validation_supported312'])==312 and len(groups['validation_unsupported76'])==76


def values(records,field):
    if field=='dhash':
        return {dhash(row[field]) for row in records if row.get(field) is not None}
    return {str(row[field]) for row in records if row.get(field) is not None}


def describe(records):
    counts=Counter(row.get('scientific_name') for row in records)
    duplicates={}
    for field in ('key','sha256','observation_id','photo_id','dhash'):
        populated=[str(row[field]) for row in records if row.get(field) is not None]
        duplicates[field]={key:n for key,n in Counter(populated).items() if n>1}
    return {'images':len(records),'species':len(counts),'photographers':len(values(records,'user_id')),
            'split_counts':dict(Counter(row.get('split','not_preserved_in_feature_rows') for row in records)),
            'cropped_images':sum(row.get('bbox') is not None for row in records),
            'accepted_flags':dict(Counter(str(row.get('accepted','not_available')) for row in records)),
            'multi_box_images':sum(len(row.get('boxes',[]))>1 for row in records),
            'per_class':dict(sorted(counts.items())), 'min_class_count':min(counts.values()),'max_class_count':max(counts.values()),
            'classes_below10':{k:n for k,n in counts.items() if n<10},
            'metadata_missing':{field:sum(row.get(field) is None for row in records) for field in ('user_id','observation_id','photo_id','sha256','dhash')},
            'duplicate_identity_or_fingerprints':duplicates,
            'missing_image_paths':[row['key'] for row in records if not Path(row['path']).is_file()]}


def overlap(left,right):
    return {field:sorted(values(left,field)&values(right,field),key=str) for field in ('key','user_id','observation_id','photo_id','sha256','dhash')}


selected_names=('train','train_expanded1577','n_val','o_val','frozen_test61','detection_test128')
overlaps={f'{left}__{right}':overlap(groups[left],groups[right]) for left,right in combinations(selected_names,2)
          if not (left=='train' and right=='train_expanded1577')}
near=[]
for train_name in ('train','train_expanded1577'):
    training=[row for row in groups[train_name] if row.get('dhash') is not None]
    for hold_name in ('n_val','o_val','frozen_test61','detection_test128'):
        matches=[]
        for a in training:
            a_hash=dhash(a['dhash'])
            for b in groups[hold_name]:
                b_hash=dhash(b.get('dhash'))
                if b_hash is not None and (distance:=(a_hash^b_hash).bit_count())<=4:
                    matches.append({'train_key':a['key'],'holdout_key':b['key'],'hamming':distance})
        near.append({'train':train_name,'holdout':hold_name,'threshold':4,'candidates':matches})

# Re-read every image fingerprint once for the proposed training/validation contract.
unique_files={row['path']:row['sha256'] for name in ('train_expanded1577','validation388') for row in groups[name]}
image_bytes=0
for path,expected in unique_files.items():
    assert sha(path)==expected,path
    image_bytes+=Path(path).stat().st_size

simple_fields=('key','path','sha256','scientific_name','target','user_id','observation_id','photo_id','dhash','bbox','accepted','split','annotation_level','source_manifest_path')
for name in ('train','train_expanded1577','validation388','validation_supported312','validation_unsupported76'):
    write(f'{name}_proposal.json',[{key:row.get(key) for key in simple_fields} for row in groups[name]])
write('class_index80.json',[{'local_id':local,'global_id':global_id,**labels[global_id]} for local,global_id in enumerate(classes)])
locked={name:{'path':str(sources[name]),'sha256':sha(sources[name]),'images':len(groups[name]),
              'role':'already-scored regression only; never epoch/LR/augmentation/threshold/model selection'}
        for name in ('frozen_test61','detection_test128')}
write('locked_test_roles.json',locked)
summary={
    'status':'completed_cpu_data_audit', 'utc':datetime.now(timezone.utc).isoformat(),
    'source_sha256':{str(path):sha(path) for path in {*source_files,*sources.values(),PACKAGE/'configs/china_labels.json'}},
    'groups':{name:describe(value) for name,value in groups.items()},
    'global_label_count':len(labels),'active_train_class_count':len(classes),'global_ids80':classes,
    'validation_supported_species':len({row['target'] for row in groups['validation_supported312']}),
    'validation_unsupported_species':len({row['target'] for row in groups['validation_unsupported76']}),
    'training_species_absent_from_validation':[labels[target]['scientific_name'] for target in classes if target not in {row['target'] for row in groups['validation388']}],
    'expanded_added_usable':len(usable),'expanded_missing_bbox':len(groups['expanded'])-len(usable),
    'overlaps':overlaps,'near_dhash_hamming4':near,
    'training_validation_image_SHA256_verified':True,'verified_unique_image_files':len(unique_files),'verified_source_bytes':image_bytes,
    'old_I_validation306_exists':True,'N_O_validation_previously_used_for_selection':True,
    'known_untouched_final_test_in_current_frozen_cohorts':False,
    'locked_tests':locked,
    'recommended_first_control':{'train':'train_expanded1577_proposal.json','validation':'validation388_proposal.json',
        'supported_train_species':80,'retain_global_label_mapping':True,'do_not_filter_validation_by_detection_or_acceptance':True,
        'preferred_models':['existing initialized backbone + same frozen global E head','last-block adaptation of identical backbone + same frozen global E head'],
        'alternative_closed80_models':['same initialized backbone + frozen encoder + linear80 head','same initialized backbone + last-block adaptation + identical linear80 head'],
        'closed80_selection_validation':'validation_supported312_proposal.json',
        'global_model_selection_validation':'validation388_proposal.json',
        'same_data_crops_class_head_seed_budget_and_selection_metric':True,
        'final_test_requires_new_photographer_disjoint_holdout_or_untouched_external_protocol':True},
    'limitations':['Observation-level source labels are weak labels for the selected primary box; multiple species may coexist.',
                   'No exhaustive capture-sequence grouping; near dHash is a heuristic and foundation pretraining overlap is unknown.',
                   'N/O are photographer/exact-hash independent of the proposed train but heavily reused development validation, not a new external test.',
                   'N/O cover97 species, only78 overlap the supported80. 80-way head selection must use fixed312-image supported subset; full388 requires a global-head/fallback protocol.',
                   '1434 and 1577 refer to usable image/crop records, not the entire physical disk image count.'],
}
write('finetune_data_audit.json',summary)
print(json.dumps({'groups':{name:{key:value[key] for key in ('images','species','photographers','cropped_images','min_class_count','max_class_count','metadata_missing')} for name,value in summary['groups'].items()},
                  'nonempty_overlaps':{name:{key:value for key,value in data.items() if value} for name,data in overlaps.items() if any(data.values())},
                  'near_candidates':[{**{key:value for key,value in item.items() if key!='candidates'},'count':len(item['candidates'])} for item in near],
                  'verified_unique_image_files':len(unique_files)},ensure_ascii=False))
