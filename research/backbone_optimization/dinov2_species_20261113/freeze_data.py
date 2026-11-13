"""Merge new observation sources, recompute common fingerprints and lock splits."""
from collections import Counter
from pathlib import Path
import hashlib
from PIL import Image, ImageOps
from collect import HERE, load, save, hash_int, historical_exclusions


if (HERE/'frozen_data/manifest.json').exists():
    raise SystemExit('Dataset already frozen.')
source = HERE/'new_data'
archive = HERE/'archive_data/manifest.json'
records = load(source/'train.json')+load(source/'test.json')+load(archive)['records']
users, observations, photos, digests, dhashes, _ = historical_exclusions()
retained, skipped = [], Counter()
for row in sorted(records,key=lambda r:hash_int('DINO-combined:'+r['photo_id'])):
    uid,oid,pid = (str(row[key]) for key in ('user_id','observation_id','photo_id'))
    if uid in users or oid in observations or pid in photos or row['sha256'] in digests:
        skipped['ID or SHA256'] += 1
        continue
    path = Path(row['path'])
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==row['sha256']
    with Image.open(path) as src:
        grid=list(ImageOps.exif_transpose(src).convert('RGB').convert('L').resize((9,8)).getdata())
    phash=int(''.join('1' if grid[y*9+x+1]>grid[y*9+x] else '0' for y in range(8) for x in range(8)),2)
    if phash in dhashes:
        skipped['exact normalized dHash'] += 1
        continue
    photos.add(pid)
    observations.add(oid)
    digests.add(row['sha256'])
    dhashes.add(phash)
    retained.append({**row,'dhash':f'{phash:016x}'})
folder = HERE/'frozen_data'
folder.mkdir(exist_ok=True)
splits = {split:[r for r in retained if r['split']==split] for split in ('train','test')}
assert not ({r['user_id'] for r in splits['train']} & {r['user_id'] for r in splits['test']})
for split,rows in splits.items():
    assert rows
    save(folder/f'{split}.json',rows)
save(folder/'manifest.json', {'splits':{k:{'images':len(v),'species':len({r['scientific_name'] for r in v}),
    'photographers':len({r['user_id'] for r in v}),'classes':dict(Counter(r['scientific_name'] for r in v)),
    'sha256':hashlib.sha256((folder/f'{k}.json').read_bytes()).hexdigest()} for k,v in splits.items()},
    'skipped':dict(skipped),'test_split_decided_by_photographer_before_model_inference':True,
    'prior_metadata_exclusions':['photographer','observation','photo','SHA256','exact normalized dHash'],
    'remaining_limits':['near duplicates not exhaustively excluded','foundation training overlap unknown','observation labels not instance ground truth']})
print('Frozen split sizes', {k:len(v) for k,v in splits.items()}, flush=True)
