"""Prepare review-only CUB training candidates. Never merge any existing split."""
from pathlib import Path
from collections import Counter,defaultdict
import json
import re
from audit_project import HERE,DATA,PACKAGE,read,save,sha

CUB=Path(r'D:\项目\鸟类识别\test02\CUB_200_2011\CUB_200_2011')
def table(name,value=str):
    return {int(line.split(' ',1)[0]):value(line.split(' ',1)[1])
            for line in (CUB/name).read_text(encoding='utf-8').splitlines() if line.strip()}
def normalized(name):return re.sub(r'[^a-z0-9]','',name.lower())

images=table('images.txt');splits=table('train_test_split.txt',int)
classes=table('classes.txt');targets=table('image_class_labels.txt',int)
boxes=table('bounding_boxes.txt',lambda text:list(map(float,text.split())))
labels=read(PACKAGE/'configs/china_labels.json')
by_english=defaultdict(list)
for i,row in enumerate(labels):
    if row.get('english_name'):by_english[normalized(row['english_name'])].append((i,row))
with_species=Counter()
for row in read(DATA/'phase_p_bioclip25_20261106/support_cache/records.json'):
    with_species[row['scientific_name']]+=1
mappings=[]
for class_id,name in classes.items():
    english=re.sub(r'^\d+\.','',name).replace('_',' ')
    matches=by_english[normalized(english)]
    mapped=matches[0] if len(matches)==1 else None
    mappings.append({'CUB_class_id':class_id,'CUB_english':english,
                     'candidate_global_species_id':mapped[0] if mapped else None,
                     'candidate_scientific_name':mapped[1]['scientific_name'] if mapped else None,
                     'mapping_basis':'unique normalized English name only; requires scientific-name/taxonomic review' if mapped else 'unmapped',
                     'training_admission':'pending_manual_mapping_and_dedup_review',
                     'in_existing_support80':bool(mapped and mapped[1]['scientific_name'] in with_species)})
lookup={r['CUB_class_id']:r for r in mappings}
candidates=[]
for image_id,path in images.items():
    if splits[image_id]!=1:continue
    match=lookup[targets[image_id]]
    if match['candidate_global_species_id'] is None:continue
    full=CUB/'images'/path
    x,y,w,h=boxes[image_id]
    candidates.append({'source':'CUB-200-2011','official_split':'train','image_id':image_id,
                       'path':str(full),'path_exists':full.is_file(),'CUB_class_id':targets[image_id],
                       'candidate_global_species_id':match['candidate_global_species_id'],
                       'candidate_scientific_name':match['candidate_scientific_name'],
                       'in_existing_support80':match['in_existing_support80'],
                       'GT_box_xyxy':[x,y,x+w,y+h],
                       'admission':'candidate_only_not_added_to_training',
                       'sequence_group_unknown':True})
save('cub_species_mapping_candidates.json',{'mappings':mappings,
     'no_silent_fuzzy_or_synonym_mapping':True,'all_mappings_need_manual_verification':True})
with (HERE/'additional_training_candidates.jsonl').open('x',encoding='utf-8') as stream:
    for row in candidates:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
save('additional_data_review.json',{
    'CUB_root':str(CUB),'metadata_sha256':{name:sha(CUB/name) for name in ['images.txt','classes.txt','train_test_split.txt','image_class_labels.txt','bounding_boxes.txt']},
    'official_images':len(images),'official_train':sum(v==1 for v in splits.values()),'official_test':sum(v==0 for v in splits.values()),
    'classes':len(classes),'missing_images':sum(not (CUB/'images'/p).is_file() for p in images.values()),
    'unique_name_matches':sum(r['candidate_global_species_id'] is not None for r in mappings),
    'matched_training_candidates':len(candidates),
    'candidates_in_existing80':sum(r['in_existing_support80'] for r in candidates),
    'matched_classes_in_existing80':sum(r['in_existing_support80'] for r in mappings),
    'official_test_added_to_candidates':False,'training_started':False,'original_manifests_modified':False,
    'new_images_downloaded':0,
    'notes':['English-name matches are proposals, not accepted scientific-name mappings',
             'Source images have bbox and part labels; bbox crops can support species-only evaluation',
             'Keep official test locked and preserve existing China validation/test roles',
             'CUB does not uniformly expose photographer/sequence IDs; near-duplicate grouping remains needed',
             'CUB train has historically been used in YOLO training; this is not a new pristine end-to-end detector benchmark']})
print('DATA_CANDIDATES',len(candidates),'classes',sum(r['candidate_global_species_id'] is not None for r in mappings),
      'existing80_images',sum(r['in_existing_support80'] for r in candidates),flush=True)
