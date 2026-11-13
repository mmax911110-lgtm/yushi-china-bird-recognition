"""Build paired feature caches on fixed crops, with test scoring unlocked by selection."""
import argparse
import time
import numpy as np
import torch
from data import HERE, PACKAGE, WORK, CLASSES, LABELS, IDS, crop, photo, read, save, save_npz, load_npz, sha, support_rows, validation_rows
from encoders import DinoEncoder, bioclip

CACHE = HERE / 'features'
WIDTH = {'bio': 768, 'dino': 384, 'bio25': 1024}


def encode(split, kind, device):
    path = CACHE / f'{split}.json'
    rows = read(path)
    dest = CACHE / f'{split}_{kind}.npz'
    if dest.exists():
        assert load_npz(dest)['rows_sha'].item() == sha(path)
        return
    model = DinoEncoder(device) if kind == 'dino' else bioclip(device, 'bioclip25' if kind == 'bio25' else 'bioclip2')
    values = np.zeros((len(rows), WIDTH[kind]), np.float32)
    present = np.array([i for i, row in enumerate(rows) if row['bbox'] is not None])
    batch = 8 if kind == 'dino' else 4 if kind == 'bio' else 2
    started = time.perf_counter()
    for begin in range(0, len(present), batch):
        indexes = present[begin:begin+batch]
        images = [crop(rows[i]) for i in indexes]
        values[indexes] = model.encode(images) if kind == 'dino' else model.encode_images(images)[0]
        if begin % 128 == 0:
            print(split, kind, begin, '/', len(present), flush=True)
    assert np.isfinite(values).all() and np.allclose(np.linalg.norm(values[present],axis=1),1,atol=.01)
    save_npz(dest, features=values, rows_sha=np.array(sha(path)))
    print(split, kind, 'done', len(present), round(time.perf_counter()-started, 2), flush=True)


def from_detector(split, device):
    from engine import BirdRecognizer
    source = HERE / 'frozen_data' / ('train.json' if split == 'expanded' else 'test.json')
    frozen = read(source.parent / 'manifest.json')
    assert sha(source) == frozen['splits'][source.stem]['sha256']
    if split == 'test':
        selection = HERE / 'runs/selection.json'
        if not selection.exists() or not read(selection)['candidates']:
            raise RuntimeError('Freeze validation selections before reading test photos.')
    target = CACHE / f'{split}.json'
    if target.exists():
        return
    pending = CACHE / f'{split}_partial.json'
    rows = read(source)
    known = {r['key']:r for r in read(pending)} if pending.exists() else {}
    worker = BirdRecognizer(PACKAGE, device=device, mode='baseline')
    try:
        for index, row in enumerate(rows):
            if row['key'] not in known:
                result = worker.predict(photo(row), detection_mode='normal')
                primary = result['primary_index']
                selected = result['boxes'][primary] if primary is not None else None
                known[row['key']] = {**row, 'target': IDS[row['scientific_name']],
                    'bbox': None if selected is None else selected['xyxy'],
                    'accepted': False if selected is None else bool(selected['species_info']['accepted']),
                    'boxes': [box['xyxy'] for box in result['boxes']]}
            if (index + 1) % 32 == 0:
                save(pending, list(known.values()))
                print(split, 'crops', index+1, '/', len(rows), flush=True)
        save(target, [known[r['key']] for r in rows])
    finally:
        worker.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['original', 'expanded', 'test'])
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--step', choices=['prepare',*WIDTH,'all'], default='all')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.phase == 'original':
        splits = ['train','n_val','o_val']
        if args.step in ('all','prepare'):
            rows = [support_rows(), validation_rows('n_val'), validation_rows('o_val')]
            assert not ({r['user_id'] for r in rows[0]} & {r['user_id'] for r in rows[1]+rows[2]})
            for split, value in zip(splits, rows):
                path = CACHE / f'{split}.json'
                if path.exists():
                    assert read(path) == value
                else:
                    save(path, value)
    else:
        splits = [args.phase]
        if args.step in ('all','prepare'):
            from_detector(args.phase, args.device)
    if args.step == 'prepare':
        return
    for kind in WIDTH if args.step == 'all' else [args.step]:
        for split in splits:
            if kind == 'bio25' and split in ('train','expanded'):
                continue
            encode(split, kind, args.device)
        if args.device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
