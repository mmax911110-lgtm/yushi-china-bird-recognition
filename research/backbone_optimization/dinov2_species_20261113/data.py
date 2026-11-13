from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageOps
from encoders import HERE, PACKAGE, read, sha
from inference.engine_n import square_context

WORK = HERE.parent
OLD = WORK / 'bioclip2_china_eval_20261028'
PHASE = OLD / 'phase_p_bioclip25_20261106'
LABELS = read(PACKAGE / 'configs/china_labels.json')
IDS = {r['scientific_name']: i for i, r in enumerate(LABELS)}
SUPPORT = read(PHASE / 'support_cache/records.json')
CLASSES = np.array(sorted({IDS[r['scientific_name']] for r in SUPPORT}), dtype=np.int64)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def support_rows():
    bindings = read(PHASE / 'support_cache/binding.json')
    assert sha(PHASE / 'support_cache/records.json') == bindings['records_sha256']
    result = []
    for row in SUPPORT:
        assert row['split'] == 'support'
        result.append({'key': row['key'], 'path': row['path'], 'sha256': row['source_sha256'],
            'user_id': row['user_id'], 'target': IDS[row['scientific_name']],
            'bbox': row['boxes'][row['primary_box_index']]['xyxy'], 'accepted': True})
    return result


def validation_rows(stage):
    prior = WORK / 'adaptive_fusion_20261113'
    source_path = PHASE / f'evaluation/{stage}/records.json'
    binding = read(prior / f'cache_binding_{stage}.json')
    assert sha(source_path) == binding['records_sha256']
    result = []
    for item in read(source_path):
        row, output = item['row'], item['output']
        primary = output['primary_index']
        box = output['boxes'][primary] if primary is not None else None
        result.append({'key': row['key'], 'path': row['path'], 'sha256': row['sha256'],
            'target': IDS[row['scientific_name']], 'user_id': row['user_id'],
            'bbox': None if box is None else box['xyxy'],
            'accepted': False if box is None else bool(box['species_info']['accepted'])})
    return result


def photo(row):
    if sha(row['path']) != row['sha256']:
        raise ValueError(f"Source image changed: {row['key']}")
    with Image.open(row['path']) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def crop(row):
    return square_context(photo(row), row['bbox'])
