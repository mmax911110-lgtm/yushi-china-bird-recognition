from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from data import CLASSES, LABELS, IDS, photo, crop, load_npz

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'dinov2_species_20261113'
PACKAGE=Path(r'D:\项目\鸟类识别\识别模型\v1\v1.1\NVIDIA')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def atomic_save(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def state_sha(module):
    return mapping_sha(module.state_dict())


def mapping_sha(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def warmstart(seed):
    path=OLD/'runs'/f"{read(HERE/'config.json')['warmstart_group']}_fusion_{seed}"/'best.pt'
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    assert checkpoint['global_ids']==CLASSES.tolist() and checkpoint['mode']=='fusion'
    return path,checkpoint


def cache(split):
    folder=HERE/'tokens'
    contract=read(folder/(split+'.json'))
    rows=read(OLD/'features'/(split+'.json'))
    assert contract['rows_sha256']==sha(OLD/'features'/(split+'.json'))
    metadata=read(folder/'manifest.json')
    assert contract['prefix_binding_sha256']==metadata['binding_sha256']
    assert metadata['source_checkpoint_sha256']==sha(OLD/'assets/dinov2_vits14_pretrain.pth')
    tokens=np.load(folder/(split+'.npy'),mmap_mode='r',allow_pickle=False)
    bio=load_npz(OLD/'features'/(split+'_bio.npz'))
    assert bio['rows_sha'].item()==contract['rows_sha256']
    assert bio['features'].shape==(len(rows),768)
    assert tokens.shape==(len(rows),257,384)
    indexes=np.array([i for i,row in enumerate(rows) if row['bbox'] is not None and row['target'] in CLASSES])
    return rows,tokens,bio['features'],indexes
