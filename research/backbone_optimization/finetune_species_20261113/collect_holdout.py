"""Bounded prediction-blind iNaturalist holdout collection and input rendering.

Reuses the public API, license filtering and SHA/dHash approach of the existing
collect_historical/collect_archive tools. Never imports a model or performs inference.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

RUNTIME=Path(sys.executable).parent
for part in ('base','primary'):
    path=RUNTIME/'packages'/part
    if path.exists():
        sys.path.insert(0,str(path))
from PIL import Image, ImageOps, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DATA=HERE/'holdout_data'
LICENSES={'cc0','cc-by','cc-by-sa','cc-by-nc','cc-by-nc-sa'}
FIELDS=('user_id','observation_id','photo_id','sha256','dhash')
SEED='Finetune-independent-holdout-20261113-v1'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def rank(value):
    return hashlib.sha256((SEED+':'+str(value)).encode()).hexdigest()


def norm(value):
    if value is None:
        return None
    if isinstance(value,int):
        return f'{value:016x}'
    text=str(value)
    return f"{int(text,2 if len(text)==64 and set(text)<={'0','1'} else 16):016x}"


def walk(value):
    if isinstance(value,dict):
        if any(k in value for k in FIELDS):
            yield value
        for sub in value.values():
            if isinstance(sub,(dict,list)):
                yield from walk(sub)
    elif isinstance(value,list):
        for sub in value:
            yield from walk(sub)


def exclusions():
    blocked_file=HERE/'data_review/blocked_identity_values.json'
    blocked=load(blocked_file)
    values={key:set(norm(v) if key=='dhash' else str(v) for v in blocked['values'][key]) for key in FIELDS}
    sources=[{'path':str(blocked_file),'sha256':sha(blocked_file)}]
    for item in blocked['source_files']:
        assert sha(item['path'])==item['sha256'],item['path']
        sources.append({'path':item['path'],'sha256':item['sha256']})
    for filename in ('train_expanded1577_proposal.json','validation388_proposal.json'):
        path=HERE/'data_review'/filename
        sources.append({'path':str(path),'sha256':sha(path)})
        for row in walk(load(path)):
            for key in FIELDS:
                value=row.get('sha256') or row.get('source_sha256') if key=='sha256' else row.get(key)
                if value is not None:
                    values[key].add(norm(value) if key=='dhash' else str(value))
    return values,sources


def fetch(url,cap):
    req=urllib.request.Request(url,headers={'User-Agent':'ChinaBirdResearch/1.1 (local independent species evaluation)'})
    with urllib.request.urlopen(req,timeout=18) as response:
        body=response.read(cap+1)
    if len(body)>cap:
        raise ValueError('response exceeds bounded byte cap')
    return body


def targets():
    supported=load(ROOT/'bioclip_dinov2_audit_20261113/active_training_classes.json')['per_class']
    catalog={row['scientific_name']:row for row in load(ROOT/'bioclip2_china_eval_20261028/china_candidate_catalog.json')}
    assert len(supported)==80
    return {name:catalog[name] for name in sorted(supported)}


def protocol(values,sources):
    return {'seed':SEED,'source':'iNaturalist research-grade observation species in China; not expert per-individual labels',
            'api_documentation':'https://api.inaturalist.org/v1/docs/',
            'query_start':'2023-01-01','query_end':'2025-12-31','query_order':'observed_on ascending',
            'target_species':list(targets()),'target_images':160,'minimum_desired_images':120,
            'preferred_images_per_species':2,'maximum_images_per_species_if_needed':4,
            'all_splits':'test','global_maximum_images_per_photographer':1,
            'licenses':sorted(LICENSES),'maximum_API_queries':120,'maximum_photo_bytes':6*1024**2,
            'query_page_size':200,'API_start_spacing_seconds':1.25,'API_workers':3,'photo_workers':4,
            'no_per_request_retry':True,'source_manifests':sources,
            'excluded_identity_counts':{key:len(value) for key,value in values.items()},
            'selection':'Prediction-blind, first pass one/two per class then bounded fill; fixed photo hashes and global photographer isolation',
            'inference_or_model_outputs_used':False,'train_or_validation_use_permitted':False,
            'source_labels_may_not_identify_every_bird':True,'foundation_pretraining_overlap_known':False}


def query(name,taxon,page,blocked,start='2023-01-01'):
    suffix='' if start=='2023-01-01' else '_'+start[:4]
    path=DATA/'sources'/f"taxon_{taxon}{suffix}_page_{page}.json"
    if path.exists():
        return load(path)
    args={'taxon_id':taxon,'place_id':6903,'quality_grade':'research','captive':'false',
          'photo_license':','.join(sorted(LICENSES)),'d1':start,'d2':'2025-12-31',
          'per_page':200,'page':page,'order_by':'observed_on','order':'asc'}
    url='https://api.inaturalist.org/v1/observations?'+urllib.parse.urlencode(args)
    result={'species':name,'taxon_id':taxon,'page':page,'query_url':url,
            'retrieved_utc':datetime.now(timezone.utc).isoformat(),'records':[],'error':None}
    try:
        data=json.loads(fetch(url,35*1024**2))
        result['total_results']=data.get('total_results')
        result['returned_observations']=len(data['results'])
        for row in data['results']:
            uid=str((row.get('user') or {}).get('id'))
            oid=str(row.get('id'))
            date=row.get('observed_on') or ''
            if uid=='None' or oid=='None' or uid in blocked['user_id'] or oid in blocked['observation_id']:
                continue
            if row.get('quality_grade')!='research' or row.get('captive') is True:
                continue
            if row.get('taxon',{}).get('id')!=taxon or not (start<=date<='2025-12-31'):
                continue
            for photo in row.get('photos',[]):
                pid=str(photo.get('id'))
                if pid=='None' or pid in blocked['photo_id'] or photo.get('license_code') not in LICENSES:
                    continue
                if not photo.get('url') or not photo.get('attribution'):
                    continue
                result['records'].append({'key':'FT_HOLDOUT_'+oid,'scientific_name':name,'split':'test',
                    'user_id':uid,'observation_id':oid,'photo_id':pid,'taxon_id':taxon,'observed_on':date,
                    'quality_grade':row['quality_grade'],'captive':row.get('captive'),
                    'observation_url':f'https://www.inaturalist.org/observations/{oid}',
                    'photo_url':photo['url'].replace('/square.','/large.'),'license':photo['license_code'],
                    'attribution':photo['attribution'],'query_source':str(path)})
                break
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
    save(path,result)
    print('QUERY',page,name,'eligible',len(result['records']),'error',bool(result['error']),flush=True)
    return result


def run_queries(jobs,blocked,start='2023-01-01'):
    results=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[]
        for job in jobs:
            futures.append(pool.submit(query,*job,blocked,start))
            time.sleep(1.25)
        for future in as_completed(futures):
            results.append(future.result())
    return results


def choose(pool,blocked,limit=160,existing_counts=None):
    groups=defaultdict(list)
    for row in pool:
        if any(str(row[key]) in blocked[key] for key in ('user_id','observation_id','photo_id')):
            continue
        groups[row['scientific_name']].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row:rank(row['photo_id']))
    chosen=[]
    taken={key:set() for key in ('user_id','observation_id','photo_id')}
    counts=Counter(existing_counts or {})
    names=sorted(groups,key=lambda name:(len(groups[name]),rank(name)))
    for quota in range(1,5):
        for name in names:
            if len(chosen)>=limit:
                return chosen
            if counts[name]>=quota:
                continue
            for row in groups[name]:
                if any(row[key] in taken[key] for key in taken):
                    continue
                chosen.append(row)
                counts[name]+=1
                for key in taken:
                    taken[key].add(row[key])
                break
    return chosen


def download(row):
    path=DATA/'photos'/(row['key']+'.jpg')
    try:
        raw=path.read_bytes() if path.exists() else fetch(row['photo_url'],6*1024**2)
        with Image.open(io.BytesIO(raw)) as opened:
            image=ImageOps.exif_transpose(opened).convert('RGB')
            if min(image.size)<160:
                raise ValueError('source shortest side is below 160 pixels')
            size=list(image.size)
            grid=list(image.convert('L').resize((9,8)).getdata())
        dhash=int(''.join('1' if grid[y*9+x+1]>grid[y*9+x] else '0' for y in range(8) for x in range(8)),2)
        return raw,hashlib.sha256(raw).hexdigest(),f'{dhash:016x}',size,None
    except Exception as error:
        return None,None,None,None,f'{type(error).__name__}: {error}'


def collect():
    if (DATA/'manifest.json').exists():
        raise SystemExit('Holdout is already frozen; do not overwrite.')
    blocked,snapshots=exclusions()
    save(DATA/'protocol.json',protocol(blocked,snapshots))
    catalog=targets()
    first_name=next(iter(catalog))
    first=query(first_name,catalog[first_name]['taxon_id'],1,blocked)
    if first['error']:
        raise SystemExit('Initial API query failed; stop without repeated network attempts: '+first['error'])
    responses=[first]+run_queries([(name,row['taxon_id'],1) for name,row in catalog.items() if name!=first_name],blocked)
    pool={row['observation_id']:row for response in responses for row in response['records']}
    chosen=choose(list(pool.values()),blocked)
    counts=Counter(row['scientific_name'] for row in chosen)
    # A single bounded additional page for at most 40 undersupplied species.
    if len(chosen)<160:
        short=sorted((name for name in catalog if counts[name]<2),key=lambda name:(counts[name],rank(name)))[:40]
        responses+=run_queries([(name,catalog[name]['taxon_id'],2) for name in short],blocked)
        pool={row['observation_id']:row for response in responses for row in response['records']}
    save(DATA/'candidates.json',{'records':list(pool.values()),'query_count':len(responses),
                                 'query_errors':[r for r in responses if r['error']]})
    # Re-read every declared source before selecting photo downloads.
    blocked,snapshots=exclusions()
    retained=[]
    rejected=[]
    attempted=set()
    remaining=dict(pool)
    for batch in range(3):
        occupied={key:blocked[key]|{r[key] for r in retained} for key in FIELDS}
        # Add retained authors globally; candidate selection does not inspect predictions.
        chosen=choose([row for oid,row in remaining.items() if oid not in attempted],occupied,160-len(retained),
                      Counter(row['scientific_name'] for row in retained))
        if not chosen:
            break
        with ThreadPoolExecutor(max_workers=4) as workers:
            for row,(raw,digest,dhash,size,error) in zip(chosen,workers.map(download,chosen)):
                attempted.add(row['observation_id'])
                if error or digest in occupied['sha256'] or dhash in occupied['dhash']:
                    rejected.append({'key':row['key'],'reason':error or 'duplicate SHA256 or exact dHash'})
                    continue
                path=DATA/'photos'/(row['key']+'.jpg')
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(raw)
                retained.append({**row,'path':str(path),'sha256':digest,'dhash':dhash,'width':size[0],'height':size[1]})
                for key in FIELDS:
                    occupied[key].add(retained[-1][key])
        print('DOWNLOADED',len(retained),'rejected',len(rejected),'batch',batch+1,flush=True)
        if len(retained)>=160:
            break
    # Final block-list recheck catches train/validation additions during network work.
    final_blocked,final_snapshots=exclusions()
    final=[]
    for row in retained:
        overlap=[key for key in FIELDS if row[key] in final_blocked[key]]
        if overlap:
            rejected.append({'key':row['key'],'reason':'final source overlap','fields':overlap})
        else:
            final.append(row)
    assert final,'No usable photos downloaded; do not freeze an empty test cohort'
    assert all(row['split']=='test' for row in final)
    assert all(len({row[key] for row in final})==len(final) for key in FIELDS)
    assert max(Counter(row['scientific_name'] for row in final).values(),default=0)<=4
    save(DATA/'test.json',final)
    summary={'status':'frozen_inputs_QA_pending','images':len(final),
             'desired_120_to_160_image_target_met':120<=len(final)<=160,
             'species':len({r['scientific_name'] for r in final}),
             'photographers':len({r['user_id'] for r in final}),
             'per_species':dict(Counter(r['scientific_name'] for r in final)),
             'source_files':final_snapshots,'test_sha256':sha(DATA/'test.json'),
             'protocol_sha256':sha(DATA/'protocol.json'),'query_count':len(responses),
             'query_errors':[{'species':r['species'],'page':r['page'],'error':r['error']} for r in responses if r['error']],
             'rejected':rejected,'all_images_are_test':True,'model_outputs_used':False,
             'training_or_hyperparameter_selection_forbidden':True,'labels_not_changed_by_predictions':True,
             'candidate_source_photographer_exclusion_counts':{key:len(v) for key,v in final_blocked.items()},
             'foundation_pretraining_overlap_known':False,'capture_sequence_isolation_guaranteed':False}
    save(DATA/'manifest.json',summary)
    print('FROZEN',len(final),'images',summary['species'],'species',flush=True)


def supplement():
    """One authorized extra window; preserve the initial frozen cohort verbatim."""
    assert not (DATA/'final_manifest.json').exists(),'Final holdout already frozen'
    original_manifest=load(DATA/'manifest.json')
    initial=load(DATA/'test.json')
    assert initial and sha(DATA/'test.json')==original_manifest['test_sha256']
    original_hashes={'initial_manifest_sha256':sha(DATA/'manifest.json'),'initial_test_sha256':sha(DATA/'test.json')}
    blocked,snapshots=exclusions()
    for row in initial:
        assert not any(row[key] in blocked[key] for key in FIELDS)
    occupied={key:blocked[key]|{r[key] for r in initial} for key in FIELDS}
    catalog=targets()
    counts=Counter(row['scientific_name'] for row in initial)
    names=sorted((name for name in catalog if counts[name]<2),key=lambda name:(counts[name],rank(name)))[:40]
    responses=[]
    if len(initial)<160 and names:
        first_name=names[0]
        first=query(first_name,catalog[first_name]['taxon_id'],1,occupied,'2024-01-01')
        responses=[first]
        # A network resolution failure stops this bounded extra window immediately.
        if not first['error'] or 'getaddrinfo' not in first['error']:
            responses+=run_queries([(name,catalog[name]['taxon_id'],1) for name in names[1:]],occupied,'2024-01-01')
    pool={r['observation_id']:r for response in responses for r in response['records']}
    save(DATA/'supplement_candidates.json',{'records':list(pool.values()),'query_count':len(responses),
        'window':'2024-2025','labels_or_model_outputs_used_for_selection':False})
    retained=list(initial)
    rejected=[]
    attempted=set()
    for batch in range(3):
        occupied={key:blocked[key]|{r[key] for r in retained} for key in FIELDS}
        chosen=choose([r for oid,r in pool.items() if oid not in attempted],occupied,160-len(retained),
                      Counter(r['scientific_name'] for r in retained))
        if not chosen:
            break
        with ThreadPoolExecutor(max_workers=4) as workers:
            for row,(raw,digest,dhash,size,error) in zip(chosen,workers.map(download,chosen)):
                attempted.add(row['observation_id'])
                if error or digest in occupied['sha256'] or dhash in occupied['dhash']:
                    rejected.append({'key':row['key'],'reason':error or 'duplicate SHA256 or exact dHash'})
                    continue
                path=DATA/'photos'/(row['key']+'.jpg')
                path.write_bytes(raw)
                value={**row,'path':str(path),'sha256':digest,'dhash':dhash,'width':size[0],'height':size[1]}
                retained.append(value)
                for key in FIELDS:
                    occupied[key].add(value[key])
        print('SUPPLEMENT_TOTAL',len(retained),'new',len(retained)-len(initial),'batch',batch+1,flush=True)
        if len(retained)>=160:
            break
    final_blocked,final_sources=exclusions()
    assert retained[:len(initial)]==initial
    assert all(not any(row[key] in final_blocked[key] for key in FIELDS) for row in retained)
    assert all(len({row[key] for row in retained})==len(retained) for key in FIELDS)
    assert all(row['split']=='test' for row in retained)
    assert 0<len(retained)<=160 and max(Counter(r['scientific_name'] for r in retained).values())<=4
    save(DATA/'final_test.json',retained)
    save(DATA/'final_manifest.json',{
        'status':'final_inputs_frozen_manual_QA_pending',**original_hashes,
        'test_sha256':sha(DATA/'final_test.json'),'images':len(retained),
        'species':len({r['scientific_name'] for r in retained}),
        'photographers':len({r['user_id'] for r in retained}),
        'photographer_identity_basis':'Observation user_id as observer/photographer proxy; attribution retained',
        'initial_images':len(initial),'additional_images':len(retained)-len(initial),
        'desired_120_to_160_image_target_met':120<=len(retained)<=160,
        'per_species':dict(Counter(r['scientific_name'] for r in retained)),
        'all_images_are_test':True,'global_maximum_images_per_observer':1,'maximum_images_per_species':4,
        'source_files':final_sources,'supplement_window':'2024-01-01 through 2025-12-31',
        'supplement_queries':len(responses),'total_API_queries':original_manifest['query_count']+len(responses),
        'supplement_query_errors':[{'species':r['species'],'error':r['error']} for r in responses if r['error']],
        'rejected':rejected,'any_model_inference_performed':False,
        'training_or_hyperparameter_selection_forbidden':True,'source_labels_changed_by_predictions':False,
        'foundation_pretraining_overlap_known':False,'capture_sequence_isolation_guaranteed':False})
    assert sha(DATA/'manifest.json')==original_hashes['initial_manifest_sha256']
    assert sha(DATA/'test.json')==original_hashes['initial_test_sha256']
    print('FINAL_FROZEN',len(retained),'images',len({r['scientific_name'] for r in retained}),'species',flush=True)


def audit(final=True):
    manifest_name='final_manifest.json' if final else 'manifest.json'
    test_name='final_test.json' if final else 'test.json'
    manifest=load(DATA/manifest_name)
    rows=load(DATA/test_name)
    assert rows,'Empty holdout cannot pass input isolation or QA'
    assert sha(DATA/test_name)==manifest['test_sha256']
    blocked,snapshot=exclusions()
    overlaps=[]
    for row in rows:
        fields=[key for key in FIELDS if row[key] in blocked[key]]
        if fields:
            overlaps.append({'key':row['key'],'fields':fields})
        assert sha(row['path'])==row['sha256']
        assert row['license'] in LICENSES and row['attribution'] and row['observation_url']
        assert row['split']=='test' and row['scientific_name'] in targets()
    assert not overlaps
    assert all(len({r[key] for r in rows})==len(rows) for key in FIELDS)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    title=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
    sheet_folder=DATA/('input_QA' if final else 'input_QA_initial')
    sheet_folder.mkdir(exist_ok=True)
    sheets=[]
    for start in range(0,len(rows),24):
        page=Image.new('RGB',(1280,1690),(246,246,246))
        draw=ImageDraw.Draw(page)
        draw.text((12,8),f'Unscored fine-tune holdout input QA {start+1}-{min(start+24,len(rows))} / {len(rows)}',font=title,fill='black')
        for position,row in enumerate(rows[start:start+24]):
            x=(position%4)*320;y=48+(position//4)*273
            with Image.open(row['path']) as original:
                image=ImageOps.exif_transpose(original).convert('RGB')
                image.thumbnail((308,215))
                page.paste(image,(x+(320-image.width)//2,y+(215-image.height)//2))
            common=targets()[row['scientific_name']].get('common_name') or row['scientific_name']
            draw.text((x+5,y+217),f"{start+position+1:03d} {row['observation_id']}",font=font,fill='black')
            draw.text((x+5,y+242),common,font=font,fill='black')
        path=sheet_folder/f'inputs_{start//24+1:02d}.jpg'
        page.save(path,quality=94)
        sheets.append({'path':str(path),'sha256':sha(path)})
    save(DATA/('isolation_audit.json' if final else 'initial_isolation_audit.json'),{'status':'isolation_passed_manual_QA_pending',
         'manifest_sha256':sha(DATA/manifest_name),'test_sha256':sha(DATA/test_name),
         'images':len(rows),'species':len({r['scientific_name'] for r in rows}),
         'photographers':len({r['user_id'] for r in rows}),'overlaps':overlaps,
         'source_files':snapshot,'image_SHA256_all_verified':True,'global_distinct_photographers':True,
         'licenses':dict(Counter(r['license'] for r in rows)),'contact_sheets':sheets,
         'any_model_inference_performed':False,'source_labels_or_denominator_changed':False,
         'foundation_pretraining_overlap_known':False,'exhaustive_near_duplicate_scan_performed':False})
    print('ISOLATION_PASS',len(rows),'images; sheets',len(sheets),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('plan','collect','supplement','audit','audit_initial'))
    args=parser.parse_args()
    DATA.mkdir(exist_ok=True)
    if args.stage=='plan':
        blocked,snapshots=exclusions()
        save(DATA/'prepared_plan.json',protocol(blocked,snapshots))
        print('PLAN',len(targets()),'species; blocked', {k:len(v) for k,v in blocked.items()},flush=True)
    else:
        {'collect':collect,'supplement':supplement,'audit':audit,'audit_initial':lambda:audit(False)}[args.stage]()
