"""Audit existing manifests, physical files and inherited split boundaries."""
from pathlib import Path
from collections import Counter,defaultdict
import json
import os
from audit_project import HERE,WORK,DATA,DATASETS,read,save as first_save,sha

IMAGE_SUFFIXES={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}


def save(name,value):
    path=HERE/name
    if path.exists():assert read(path)==value,'Audit output differs; use a new revision: '+name
    else:first_save(name,value)


def extract(value):
    if isinstance(value,list):
        return [r for r in value if isinstance(r,dict) and ('path' in r or 'scientific_name' in r or 'photo_id' in r)]
    if isinstance(value,dict):
        for key in ['records','rows','images']:
            if isinstance(value.get(key),list):return extract(value[key])
    return []


def norm_hash(value):
    if value is None:return None
    try:
        s=str(value)
        return int(s,2) if len(s)==64 and set(s)<=set('01') else int(s,16)
    except ValueError:return None


def describe(path,rows):
    count=Counter(r['scientific_name'] for r in rows if r.get('scientific_name'))
    label_basis='scientific_name field; unlabeled/negative rows excluded from species count'
    if path.name=='cub_manifest.json':
        count=Counter(f"CUB_class_{r['class_id']:03d}" for r in rows)
        label_basis='CUB official class_id; not the current global species ID'
    splits=Counter(r.get('split','unspecified') for r in rows)
    return {'path':str(path),'sha256':sha(path),'rows':len(rows),'species_count':len(count),
            'splits':dict(splits),'per_class':dict(sorted(count.items())), 'label_basis':label_basis,
            'rows_without_scientific_name':sum(not r.get('scientific_name') for r in rows),
            'records_with_SHA':sum(bool(r.get('sha256',r.get('source_sha256'))) for r in rows),
            'records_with_dHash':sum(r.get('dhash') is not None for r in rows),
            'photographers':len({r['user_id'] for r in rows if r.get('user_id') is not None}),
            'missing_declared_image_paths':sum(bool(r.get('path')) and not Path(r['path']).is_file() for r in rows)}


def overlap(left,right):
    fields=['user_id','observation_id','photo_id','sha256','dhash']
    result={}
    for field in fields:
        def values(rows):
            if field=='sha256':return {r.get('sha256',r.get('source_sha256')) for r in rows if r.get('sha256',r.get('source_sha256'))}
            if field=='dhash':return {h for r in rows if (h:=norm_hash(r.get('dhash'))) is not None}
            return {str(r[field]) for r in rows if r.get(field) is not None}
        a,b=values(left),values(right)
        common=a&b
        result[field]={'left_with_unique_value':len(a),'right_with_unique_value':len(b),'overlap_unique_values':len(common)}
    return result


def main():
    relative=['china_images.json','cub_manifest.json','iber_manifest.json',
              'phase_b/test_manifest.json','phase_c/manifest.json','phase_d/manifest.json','phase_e/manifest.json',
              'phase_f/manifest.json','phase_g/manifest.json','phase_i/manifest.json',
              'phase_l_mobile_20261106/dataset.json','phase_m_confusion_20261106/manifest.json',
              'phase_n_accuracy_20261106/fresh_data/manifest.json','phase_o_offline_accuracy_20261106/fresh_data/manifest.json',
              'phase_p_bioclip25_20261106/fresh_data/manifest.json','phase_p_bioclip25_20261106/support_cache/records.json']
    files=[DATA/name for name in relative]
    files += [WORK/f'{folder}/fresh_test/manifest.json' for folder in ['nonlinear_fusion_20261113','amu_fusion_20261113']]
    manifests=[];all_rows={};missing=[]
    for path in files:
        if not path.exists():missing.append(str(path));continue
        rows=extract(read(path))
        key=path.relative_to(DATA).as_posix() if path.is_relative_to(DATA) else path.relative_to(WORK).as_posix()
        manifests.append(describe(path,rows));all_rows[key]=rows
    save('manifest_audit_v2.json',{'manifests':manifests,'missing_optional_paths':missing})
    # A data directory may hold originals, derived crops and figures together.
    # Count physical files separately; never call their sum unique training data.
    root=DATASETS/'02_中国鸟种识别'
    images=[];groups=Counter();extensions=Counter()
    for folder,dirs,files_here in os.walk(root,followlinks=False):
        for name in files_here:
            p=Path(folder)/name
            if p.suffix.lower() not in IMAGE_SUFFIXES:continue
            rel=p.relative_to(root)
            stage=rel.parts[1] if len(rel.parts)>2 else rel.parts[0]
            groups[stage]+=1;extensions[p.suffix.lower()]+=1
            images.append(str(p))
    save('physical_species_images.json',{'root':str(root),'physical_image_files':len(images),
         'counts_by_stage_directory':dict(groups),'extensions':dict(extensions),
         'warning':'Includes original photos, crops, figures and historical repeated images; not a unique labeled-training count'})
    c=all_rows.get('phase_c/manifest.json',[])
    i=all_rows.get('phase_i/manifest.json',[])
    active=all_rows.get('phase_p_bioclip25_20261106/support_cache/records.json',[])
    within={}
    for name,rows in [('C',c),('I',i)]:
        splits=sorted({r.get('split','unspecified') for r in rows})
        within[name]={f'{a}_vs_{b}':overlap([r for r in rows if r.get('split')==a],
                                          [r for r in rows if r.get('split')==b])
                      for j,a in enumerate(splits) for b in splits[j+1:]}
    against={name:overlap(active,rows) for name,rows in all_rows.items()
             if '/fresh_data/manifest.json' in name or name.endswith('fresh_test/manifest.json')}
    counts=Counter(r['scientific_name'] for r in active)
    save('active_training_classes.json',{'images':len(active),'classes':len(counts),
         'per_class':dict(sorted(counts.items())),'min_images':min(counts.values()),'max_images':max(counts.values()),
         'classes_below10':[{'scientific_name':k,'images':v} for k,v in sorted(counts.items()) if v<10]})
    save('split_overlap_audit.json',{'within_original_manifests':within,'active_support_vs_recent_frozen_sets':against,
          'scope':'Metadata SHA/exact normalized dHash, photographers and observation IDs; not a fresh perceptual-near-duplicate image scan',
          'sequence_limitation':'user_id/observation_id grouping is available for iNaturalist. Burst/camera sequence metadata not uniformly available; unknown is not zero overlap.'})
    unique_sha={r.get('sha256',r.get('source_sha256')) for rows in all_rows.values() for r in rows if r.get('sha256',r.get('source_sha256'))}
    print('MANIFESTS',len(manifests),'ROWS',sum(r['rows'] for r in manifests),'UNIQUE_SHA_REPORTED',len(unique_sha),'PHYSICAL_IMAGES',len(images),flush=True)
    print('ACTIVE_SUPPORT',len(active),len(counts),min(counts.values()),max(counts.values()),flush=True)
    print('SUMMARIES',[(Path(r['path']).parent.name,Path(r['path']).name,r['rows'],r['species_count'],r['splits']) for r in manifests],flush=True)


if __name__=='__main__':main()
