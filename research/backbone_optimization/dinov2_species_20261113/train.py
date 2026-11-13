"""Train the species-only heads on cached frozen embeddings."""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data import HERE, CLASSES, read, save, load_npz, sha
from fusion import SpeciesNet

CACHE = HERE / 'features'


def load(split, kind):
    rows = CACHE / f'{split}.json'
    data = load_npz(CACHE / f'{split}_{kind}.npz')
    assert data['rows_sha'].item() == sha(rows)
    return data['features']


def split_arrays(split):
    rows = read(CACHE / f'{split}.json')
    targets = np.array([r['target'] for r in rows])
    usable = np.array([r['bbox'] is not None for r in rows]) & np.isin(targets, CLASSES)
    return tuple(torch.from_numpy(x[usable]) for x in (load(split,'bio'), load(split,'dino'), np.searchsorted(CLASSES, targets)))


def fit(mode, group, seed, config, device, resume=False):
    output = HERE / 'runs' / f'{group}_{mode}_{seed}'
    output.mkdir(parents=True, exist_ok=True)
    done = output / 'complete.json'
    if done.exists():
        return read(done)
    train_splits = ['train'] + (['expanded'] if group == 'expanded' else [])
    bindings = {name:sha(CACHE / name) for split in train_splits + config['validation']
        for name in (f'{split}.json',f'{split}_bio.npz',f'{split}_dino.npz')}
    contract = {'mode': mode, 'group': group, 'seed': seed, 'config': config, 'input_hashes': bindings,
                'global_ids': CLASSES.tolist(), 'device': device}
    if (output/'contract.json').exists():
        assert read(output/'contract.json') == contract
    else:
        save(output/'contract.json',contract)
    train_arrays = [split_arrays(split) for split in train_splits]
    x,y,target = (torch.cat(parts).to(device) for parts in zip(*train_arrays))
    val = {split: tuple(t.to(device) for t in split_arrays(split)) for split in config['validation']}
    torch.manual_seed(seed)
    order = torch.Generator().manual_seed(seed)
    model = SpeciesNet(mode,len(CLASSES),config['projection'],config['dropout']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,config['epochs'],eta_min=config['learning_rate']/10)
    history, best = [], (-1, -math.inf)
    last = output/'last.pt'
    if last.exists():
        if not resume:
            raise RuntimeError(f'Incomplete run {output.name}; use --resume.')
        state = torch.load(last, map_location='cpu', weights_only=True)
        assert state['contract_hash'] == sha(output/'contract.json')
        model.load_state_dict(state['model'],strict=True)
        opt.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        order.set_state(state['shuffle_rng'])
        torch.set_rng_state(state['cpu_rng'])
        if device == 'cuda':
            torch.cuda.set_rng_state(state['cuda_rng'])
        history, best = state['history'], tuple(state['best'])
    counts = torch.bincount(target,minlength=len(CLASSES)).float()
    weights = counts.rsqrt()
    weights /= weights.mean()
    for epoch in range(len(history)+1,config['epochs']+1):
        model.train()
        total = 0.
        permutation = torch.randperm(len(target),generator=order).to(device)
        for index in permutation.split(config['batch_size']):
            loss = F.cross_entropy(model(x[index],y[index]),target[index],weight=weights,label_smoothing=.025)
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())*len(index)
        model.eval()
        with torch.no_grad():
            logits = {s:model(a,b) for s,(a,b,_) in val.items()}
            correct = {s:int((p.argmax(1)==val[s][2]).sum()) for s,p in logits.items()}
            loss_val = sum(float(F.cross_entropy(p,val[s][2],reduction='sum')) for s,p in logits.items())/sum(len(v[2]) for v in val.values())
        history.append({'epoch':epoch,'train_loss':total/len(target),'validation_loss':loss_val,
                        'validation_correct':sum(correct.values()),**correct})
        score = (sum(correct.values()),-loss_val)
        checkpoint = {'model':model.state_dict(),'mode':mode,'global_ids':CLASSES.tolist(), 'epoch':epoch,
                      'projection':config['projection'],'dropout':config['dropout'],'contract_hash':sha(output/'contract.json')}
        if score > best:
            best = score
            atomic_save(output/'best.pt',checkpoint)
        scheduler.step()
        atomic_save(last,{**checkpoint,'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'history':history,
                         'best':best,'shuffle_rng':order.get_state(),'cpu_rng':torch.get_rng_state(),
                         'cuda_rng':torch.cuda.get_rng_state() if device=='cuda' else None})
    with (output/'results.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    result = {'mode':mode,'group':group,'seed':seed,'best_correct':best[0],'train_images':len(target),
              'validation_with_target_crops':sum(len(v[2]) for v in val.values()),
              'checkpoint':str(output/'best.pt'),'checkpoint_sha256':sha(output/'best.pt'),
              'trainable_parameters':sum(p.numel() for p in model.parameters())}
    save(done,result)
    print(output.name,result['train_images'],result['best_correct'],flush=True)
    return result


def atomic_save(path, obj):
    tmp=path.with_suffix('.tmp')
    torch.save(obj,tmp)
    tmp.replace(path)


def load_head(path, device='cpu'):
    state = torch.load(path,map_location=device,weights_only=True)
    assert state['global_ids']==CLASSES.tolist()
    model = SpeciesNet(state['mode'],len(CLASSES),state['projection'],state['dropout']).to(device)
    model.load_state_dict(state['model'],strict=True)
    return model.eval()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group',choices=['original','expanded'],required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if not read(HERE/'smoke_result.json')['passed']:
        raise RuntimeError('GPU smoke test must pass first')
    if (HERE/'runs/selection.json').exists():
        raise RuntimeError('This experiment is already frozen.')
    config=read(HERE/'config.json')
    for mode in ('bio','dino','fusion'):
        for seed in config['seeds']:
            fit(mode,args.group,seed,config,args.device,args.resume)
