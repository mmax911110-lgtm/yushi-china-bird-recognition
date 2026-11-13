"""Revisit unselected public observation metadata from earlier collection caches."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import io
import json
import zipfile
from PIL import Image, ImageOps
from collect import HERE, ROOT, OLD, LICENSES, load, save, fetch, hash_int, historical_exclusions


def main():
    destination = HERE / 'archive_data'
    destination.mkdir(exist_ok=True)
    if (destination/'manifest.json').exists():
        raise SystemExit('Archive dataset already frozen.')
    users, observations, photos, hashes, dhashes, _ = historical_exclusions()
    targets = load(ROOT/'bioclip_dinov2_audit_20261113/active_training_classes.json')['per_class']
    catalog = {str(r['taxon_id']):r['scientific_name'] for r in load(OLD/'china_candidate_catalog.json') if r['scientific_name'] in targets}
    pool = defaultdict(dict)

    def inspect(file, value):
        parts = Path(file).stem.split('_')
        if len(parts) < 3 or parts[1] not in catalog:
            return
        taxon = parts[1]
        for row in value.get('results',[]):
            uid = str((row.get('user') or {}).get('id'))
            oid = str(row.get('id'))
            if uid == 'None' or oid == 'None' or uid in users or oid in observations or str((row.get('taxon') or {}).get('id')) != taxon:
                continue
            if row.get('quality_grade') != 'research' or row.get('captive') is True:
                continue
            available = [p for p in row.get('photos',[]) if p.get('url') and p.get('license_code') in LICENSES and str(p['id']) not in photos]
            if not available:
                continue
            photo = available[0]
            pid = str(photo['id'])
            split = 'test' if hash_int('DINO-observer:'+uid)%100<33 else 'train'
            name = catalog[taxon]
            pool[name][oid] = {'key':'DINO_'+oid, 'scientific_name':name,'split':split,'user_id':uid,
                'observation_id':oid,'photo_id':pid,'taxon_id':int(taxon),'observed_on':row.get('observed_on'),
                'observation_url':f'https://www.inaturalist.org/observations/{oid}',
                'photo_url':photo['url'].replace('/square.','/large.'),'license':photo['license_code'],
                'attribution':photo.get('attribution')}

    sources = [OLD/f'phase_{stage}_20261106/fresh_data/sources' for stage in ('n_accuracy','o_offline_accuracy','p_bioclip25')]
    sources.append(ROOT/'nonlinear_fusion_20261113/fresh_test/sources')
    archive = ROOT/'amu_fusion_20261113/fresh_test/public_source_responses.zip'
    if archive.exists():
        with zipfile.ZipFile(archive) as cache:
            for member in cache.namelist():
                if member.split('_')[1] in catalog:
                    with cache.open(member) as stream:
                        inspect(member,json.load(stream))
    for folder in sources:
        for path in sorted(folder.glob('historical_*.json')):
            if path.stem.split('_')[1] in catalog:
                inspect(path.name,load(path))
    selected = []
    for name, observations_pool in sorted(pool.items()):
        seen, counts = set(), {'train':0,'test':0}
        quota = {'train':18,'test':6}
        for row in sorted(observations_pool.values(),key=lambda row:hash_int('DINO-archive:'+row['photo_id'])):
            split, uid = row['split'], row['user_id']
            if uid in seen or counts[split] >= quota[split]:
                continue
            selected.append(row)
            seen.add(uid)
            counts[split] += 1
    save(destination/'candidates.json',selected)
    print('Metadata candidates:',len(selected),flush=True)
    images=destination/'photos'
    images.mkdir(exist_ok=True)

    def download(row):
        try:
            raw=fetch(row['photo_url'],6*1024**2)
            with Image.open(io.BytesIO(raw)) as opened:
                image=ImageOps.exif_transpose(opened).convert('RGB')
                if min(image.size)<160:
                    raise ValueError('source too small')
                grid=list(image.convert('L').resize((9,8)).getdata())
            perceptual=int(''.join('1' if grid[y*9+x+1]>grid[y*9+x] else '0' for y in range(8) for x in range(8)),2)
            return raw,hashlib.sha256(raw).hexdigest(),perceptual,None
        except Exception as error:
            return None,None,None,str(error)

    records, errors = [], []
    with ThreadPoolExecutor(max_workers=6) as workers:
        for i,(row,(raw,digest,dhash,error)) in enumerate(zip(selected,workers.map(download,selected))):
            if error or digest in hashes or dhash in dhashes or row['photo_id'] in photos:
                errors.append({'key':row['key'],'reason':error or 'exact duplicate'})
                continue
            path=images/(row['key']+'.jpg')
            path.write_bytes(raw)
            records.append({**row,'path':str(path),'sha256':digest,'dhash':f'{dhash:016x}'})
            hashes.add(digest)
            dhashes.add(dhash)
            photos.add(row['photo_id'])
            if i%50==0:
                print('Downloads processed',i+1,'/',len(selected),flush=True)
    save(destination/'manifest.json',{'records':records,'errors':errors,'metadata_origin':'Earlier official iNaturalist API caches, previously selected rows and observers excluded'})
    print('Archive retained',len(records),flush=True)


if __name__=='__main__':
    main()
