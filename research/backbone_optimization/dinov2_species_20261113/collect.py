"""Collect prediction-blind China bird observations for training and final testing."""
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import urllib.parse
import urllib.request

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT / 'bioclip2_china_eval_20261028'
DATA = HERE / 'new_data'
DATA.mkdir(exist_ok=True)
LICENSES = {'cc0', 'cc-by', 'cc-by-sa', 'cc-by-nc', 'cc-by-nc-sa'}


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def fetch(url, cap=24 * 1024**2):
    req = urllib.request.Request(url, headers={'User-Agent': 'ChinaBirdResearch/1.0 (local bird species evaluation)'})
    with urllib.request.urlopen(req, timeout=18) as response:
        value = response.read(cap + 1)
    if len(value) > cap:
        raise ValueError('Response exceeds size cap')
    return value


def hash_int(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'big')


def canonical_dhash(value):
    if value is None:
        return None
    value = str(value)
    try:
        return int(value, 2 if len(value) == 64 and set(value) <= {'0', '1'} else 16)
    except ValueError:
        return None


def historical_exclusions():
    prior = load(ROOT / 'bioclip_dinov2_audit_20261113/manifest_audit_v2.json')['manifests']
    paths = {Path(row['path']) for row in prior}
    for path in tuple(paths):
        for name in ('metadata.json', 'bird_metadata.json'):
            if (path.parent / name).exists():
                paths.add(path.parent / name)
    paths.add(OLD / 'china_metadata.json')
    users, observations, photos, hashes, dhashes = set(), set(), set(), set(), set()

    def walk(value):
        if isinstance(value, list):
            for row in value:
                walk(row)
        elif isinstance(value, dict):
            for name, table in [('user_id', users), ('observation_id', observations), ('photo_id', photos)]:
                if value.get(name) is not None:
                    table.add(str(value[name]))
            if digest := value.get('sha256', value.get('source_sha256')):
                hashes.add(digest)
            if (digest := canonical_dhash(value.get('dhash'))) is not None:
                dhashes.add(digest)
            for name in ('records', 'rows', 'images', 'support', 'train', 'validation', 'test'):
                if name in value:
                    walk(value[name])

    for path in sorted(paths):
        walk(load(path))
    return users, observations, photos, hashes, dhashes, len(paths)


def main():
    if (DATA / 'manifest.json').exists():
        raise SystemExit('Dataset already frozen.')
    users, observations, photos, hashes, dhashes, prior_files = historical_exclusions()
    catalog = {r['scientific_name']: r for r in load(OLD / 'china_candidate_catalog.json')}
    targets = sorted(load(ROOT / 'bioclip_dinov2_audit_20261113/active_training_classes.json')['per_class'])
    protocol = {'seed': 'DINO-species-01', 'classes': targets, 'train_quota': 12, 'test_quota': 4,
                'query_start': '2025-01-01', 'query_end': '2026-11-13', 'source_manifests': prior_files,
                'excluded_counts': [len(x) for x in (users, observations, photos, hashes, dhashes)],
                'test_groups': 'photographer SHA256 bucket < 33 out of 100',
                'selection': 'chronological eligibility and fixed hash order, one photo per species and observer per split, before any inference',
                'source': 'iNaturalist Research Grade China, observation labels', 'model_outputs_used': False}
    save(DATA / 'protocol.json', protocol)
    collected, errors = [], []
    for index, name in enumerate(targets):
        taxon = catalog[name]['taxon_id']
        args = {'taxon_id': taxon, 'place_id': 6903, 'quality_grade': 'research', 'captive': 'false',
                'photo_license': ','.join(sorted(LICENSES)), 'd1': protocol['query_start'], 'd2': protocol['query_end'],
                'per_page': 160, 'order_by': 'observed_on', 'order': 'desc'}
        pool = []
        try:
            body = json.loads(fetch('https://api.inaturalist.org/v1/observations?' + urllib.parse.urlencode(args), 55 * 1024**2))
            for row in body['results']:
                uid = str(row['user']['id'])
                oid = str(row['id'])
                if uid in users or oid in observations or row.get('taxon', {}).get('id') != taxon:
                    continue
                for photo in row.get('photos', [])[:1]:
                    pid = str(photo['id'])
                    if pid in photos or photo.get('license_code') not in LICENSES or not photo.get('url'):
                        continue
                    split = 'test' if hash_int('DINO-observer:' + uid) % 100 < 33 else 'train'
                    pool.append({'key': 'DINO_' + oid, 'scientific_name': name, 'split': split,
                                 'user_id': uid, 'observation_id': oid, 'photo_id': pid,
                                 'observed_on': row.get('observed_on'), 'taxon_id': taxon,
                                 'observation_url': f'https://www.inaturalist.org/observations/{oid}',
                                 'photo_url': photo['url'].replace('/square.', '/large.'),
                                 'license': photo['license_code'], 'attribution': photo.get('attribution')})
        except Exception as error:
            errors.append({'species': name, 'error': str(error)})
        seen = set()
        counts = {'train': 0, 'test': 0}
        for row in sorted(pool, key=lambda r: hash_int('DINO-photo:' + r['photo_id'])):
            key = row['user_id']
            split = row['split']
            if key in seen or counts[split] >= protocol[split + '_quota']:
                continue
            seen.add(key)
            counts[split] += 1
            collected.append(row)
        print(index + 1, name, counts, 'total', len(collected), flush=True)
        time.sleep(1.15)
    save(DATA / 'candidates.json', {'records': collected, 'errors': errors})
    photo_dir = DATA / 'photos'
    photo_dir.mkdir(exist_ok=True)

    def download(row):
        try:
            raw = fetch(row['photo_url'], 6 * 1024**2)
            with Image.open(io.BytesIO(raw)) as image:
                rgb = ImageOps.exif_transpose(image).convert('RGB')
                if min(rgb.size) < 160:
                    raise ValueError('Small source image')
                grid = list(rgb.convert('L').resize((9, 8)).getdata())
            dhash = int(''.join('1' if grid[y*9+x+1] > grid[y*9+x] else '0' for y in range(8) for x in range(8)), 2)
            digest = hashlib.sha256(raw).hexdigest()
            return raw, digest, dhash, None
        except Exception as error:
            return None, None, None, str(error)

    retained, blocked = [], []
    with ThreadPoolExecutor(max_workers=5) as pool:
        for row, (raw, digest, dhash, error) in zip(collected, pool.map(download, collected)):
            if error or digest in hashes or dhash in dhashes or row['photo_id'] in photos:
                blocked.append({'key': row['key'], 'reason': error or 'duplicate identifier or exact hash'})
                continue
            hashes.add(digest)
            dhashes.add(dhash)
            photos.add(row['photo_id'])
            file = photo_dir / (row['key'] + '.jpg')
            file.write_bytes(raw)
            retained.append({**row, 'path': str(file), 'sha256': digest, 'dhash': f'{dhash:016x}'})
            if len(retained) % 50 == 0:
                print('Downloaded', len(retained), flush=True)
    splits = {split: [r for r in retained if r['split'] == split] for split in ('train', 'test')}
    assert not ({r['user_id'] for r in splits['train']} & {r['user_id'] for r in splits['test']})
    for split, rows in splits.items():
        save(DATA / f'{split}.json', rows)
    save(DATA / 'manifest.json', {'counts': {k:len(v) for k,v in splits.items()},
                                  'classes': {k:len({r['scientific_name'] for r in v}) for k,v in splits.items()},
                                  'blocked': blocked, 'query_errors': errors, 'protocol': protocol})


if __name__ == '__main__':
    main()
