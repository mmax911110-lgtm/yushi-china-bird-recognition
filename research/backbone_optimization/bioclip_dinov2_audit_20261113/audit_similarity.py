"""Metadata near-hash risk screen, with explicit coverage and no split edits."""
from collections import Counter
from pathlib import Path
import numpy as np
from audit_project import HERE,WORK,DATA,read,save
from audit_data import norm_hash,overlap,extract

c=read(DATA/'phase_c/manifest.json')['records']
i=read(DATA/'phase_i/manifest.json')['records']
source={r['key']:r for r in c+i}
active=read(DATA/'phase_p_bioclip25_20261106/support_cache/records.json')
enriched=[]
for r in active:
    original=source[r['key']]
    assert r['source_sha256']==original['sha256']
    enriched.append(original)
held=[r for r in c+i if r['split']!='support']
for stage in ['n_accuracy','o_offline_accuracy','p_bioclip25']:
    held+=extract(read(DATA/f'phase_{stage}_20261106/fresh_data/manifest.json'))
for folder in ['nonlinear_fusion_20261113','amu_fusion_20261113']:
    held+=extract(read(WORK/f'{folder}/fresh_test/manifest.json'))


def unique(rows):
    result={}
    for r in rows:
        key=r.get('sha256',r.get('source_sha256')) or r['key']
        result.setdefault(key,r)
    return list(result.values())


def screen(training,holdout):
    train=[r for r in unique(training) if norm_hash(r.get('dhash')) is not None]
    test=[r for r in unique(holdout) if norm_hash(r.get('dhash')) is not None]
    exact=overlap(training,holdout)
    if not train or not test:
        return {'metadata_overlap':exact,'near_scan_performed':False}
    th=np.asarray([norm_hash(r['dhash']) for r in train],np.uint64)
    hh=np.asarray([norm_hash(r['dhash']) for r in test],np.uint64)
    pairs=[]
    for first in range(0,len(th),128):
        distance=np.bitwise_count(np.bitwise_xor(th[first:first+128,None],hh[None,:]))
        aa,bb=np.nonzero(distance<=4)
        for a,b in zip(aa,bb):
            x,y=train[first+int(a)],test[int(b)]
            pairs.append({'training_key':x['key'],'holdout_key':y['key'],
                          'hamming_distance':int(distance[a,b]),'training_species':x.get('scientific_name'),
                          'holdout_species':y.get('scientific_name'),'training_path':x.get('path'),
                          'holdout_path':y.get('path')})
    return {'metadata_overlap':exact,'training_dhash_rows':len(train),'holdout_dhash_rows':len(test),
            'near_scan_performed':True,'threshold':4,'pairs':pairs,'flagged_pairs':len(pairs),
            'interpretation':'Risk flags only; low-information sky/water backgrounds can collide. Not confirmed duplicate leakage.'}

results={'E_BioCLIP2_C736':screen([r for r in c if r['split']=='support'],held),
         'BioCLIP25_P1434':screen(enriched,held)}
save('similarity_risk_audit.json',{
    'method':'64-bit dHash Hamming<=4 screening from existing metadata; exact SHA/IDs also checked',
    'P_support_missing_dhash_enriched_by_verified_C_I_key_and_source_SHA':True,
    'holdout_rows_before_unique':len(held),'results':results,
    'splits_or_labels_changed':False,'not_a_complete_near_duplicate_guarantee':True,
    'sequence_metadata_uniformly_available':False})
print('NEAR_HASH_SCREEN', {k:v.get('flagged_pairs') for k,v in results.items()},flush=True)
