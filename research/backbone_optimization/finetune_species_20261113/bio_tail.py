"""Independent BioCLIP2 last-block/ln_post/proj experiment; frozen DINO CLS.

Public scoring API: restore(path,device)->(net,state),
net(bio_prefix,dino_features), forward_bio(net,bio_prefix,dino_features),
cache(split)->(rows,prefix_mmap,dino_features,supported_indexes).
All original files are read-only. prepare_test() requires a frozen selection.
"""
import argparse
from copy import deepcopy
import csv
from functools import lru_cache
import math
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import HERE, OLD, PACKAGE, CLASSES, read, save, sha, state_sha, mapping_sha, crop, load_npz, warmstart, atomic_save
from fusion import SpeciesNet
from inference.encoders import LocalEncoder

FOLDER = HERE/'bio_tokens'
CONFIG = HERE/'bio_config.json'


@lru_cache(maxsize=1)
def source_binding():
    cfg=read(PACKAGE/'configs/recognizer.json')
    weight=PACKAGE/'models/bioclip2/open_clip_model.safetensors'
    actual=sha(weight)
    assert actual==cfg['asset_hashes']['models/bioclip2/open_clip_model.safetensors']
    import open_clip.transformer as implementation
    return {'source_checkpoint':str(weight),'source_checkpoint_sha256':actual,
            'open_clip_config_sha256':sha(PACKAGE/'models/bioclip2/open_clip_config.json'),
            'encoder_source_sha256':sha(PACKAGE/'app/inference/encoders.py'),
            'transformer_source':str(Path(implementation.__file__).resolve()),
            'transformer_source_sha256':sha(implementation.__file__)}


def encoder(device):
    source_binding()
    return LocalEncoder(PACKAGE,'bioclip2',device,4,'bf16')


def assert_visual(v):
    assert len(v.transformer.resblocks)==24 and v.transformer.batch_first
    assert v.attn_pool is None and v.pool_type=='tok' and not v.final_ln_after_pool
    assert not v.output_tokens and v.proj is not None and tuple(v.proj.shape)==(1024,768)
    assert tuple(v.positional_embedding.shape)==(257,1024)


class BioTailSpeciesNet(nn.Module):
    def __init__(self,v,head_state,train_tail):
        super().__init__()
        assert_visual(v)
        self.block=deepcopy(v.transformer.resblocks[-1])
        self.ln_post=deepcopy(v.ln_post)
        self.proj=nn.Parameter(v.proj.detach().clone(),requires_grad=train_tail)
        self.attn_pool=None
        self.final_ln_after_pool=v.final_ln_after_pool
        self.pool_type=v.pool_type
        # Reuse the installed implementation's pooling, rather than guess its LN order.
        self._pool_impl=type(v)._pool
        self._global_pool_impl=type(v)._global_pool
        self.head=SpeciesNet('fusion',len(CLASSES),head_state['projection'],head_state['dropout'])
        self.head.load_state_dict(head_state['model'],strict=True)
        self.block.requires_grad_(train_tail)
        self.ln_post.requires_grad_(train_tail)
        self.head.requires_grad_(True)
        self.train_tail=train_tail

    def _global_pool(self,x):
        return self._global_pool_impl(self,x)

    def visual(self,tokens):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=tokens.is_cuda):
            # _embeds/residual tokens are BF16 in the original CUDA encoder.
            x=tokens.to(torch.bfloat16) if tokens.is_cuda else tokens.float()
            x=self.block(x,attn_mask=None)
            pooled,_=self._pool_impl(self,x)
            value=F.normalize(pooled @ self.proj,dim=-1)
        return value.float()

    def forward(self,bio_prefix,dino_features):
        return self.head(self.visual(bio_prefix),dino_features.float())


def build(seed,train_tail,device='cuda'):
    original=encoder('cpu')
    _,head_state=warmstart(seed)
    result=BioTailSpeciesNet(original.model.visual,head_state,train_tail).to(device)
    original.close()
    return result


def restore(path,device='cuda'):
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['architecture']=='bioclip_tail' and state['global_ids']==CLASSES.tolist()
    assert state['source_checkpoint_sha256']==source_binding()['source_checkpoint_sha256']
    model=build(state['seed'],state['train_tail'],device)
    model.load_state_dict(state['model'],strict=True)
    return model.eval(),state


def forward_bio(net,bio_prefix,dino_frozen_features):
    return net(bio_prefix,dino_frozen_features)


def cache(split):
    root=HERE/'features' if split=='test' else OLD/'features'
    rows_path=root/(split+'.json')
    rows=read(rows_path)
    info=read(FOLDER/(split+'.json'))
    manifest=read(FOLDER/'manifest.json')
    assert info['rows_sha256']==sha(rows_path)
    assert info['prefix_binding_sha256']==manifest['binding_sha256']
    dino=load_npz(root/(split+'_dino.npz'))
    assert dino['rows_sha'].item()==info['rows_sha256']
    assert dino['features'].shape==(len(rows),384)
    tokens=np.load(FOLDER/(split+'.npy'),mmap_mode='r',allow_pickle=False)
    assert tokens.shape==(len(rows),257,1024) and tokens.dtype==np.float32
    supported=np.array([i for i,r in enumerate(rows) if r['bbox'] is not None and r.get('target') in CLASSES],dtype=np.int64)
    return rows,tokens,dino['features'],supported


def _prepare_one(split,rows_path,original,binding_sha,device,test_selection=None):
    destination=FOLDER/(split+'.npy')
    metadata=FOLDER/(split+'.json')
    if metadata.exists():
        info=read(metadata)
        assert info['rows_sha256']==sha(rows_path) and sha(destination)==info['tokens_sha256']
        assert info['prefix_binding_sha256']==binding_sha
        if test_selection is not None:
            assert info['selection_sha256']==test_selection
        return info
    rows=read(rows_path)
    pending=destination.with_suffix('.pending.npy')
    output=np.lib.format.open_memmap(pending,mode='w+',dtype=np.float32,shape=(len(rows),257,1024))
    output[:]=0
    present=[i for i,r in enumerate(rows) if r['bbox'] is not None]
    old_path=rows_path.parent/(split+'_bio.npz')
    baseline=load_npz(old_path) if old_path.exists() else None
    if baseline is not None:
        assert baseline['rows_sha'].item()==sha(rows_path)
    maximum=0.
    dtype=None
    v=original.model.visual
    config=read(CONFIG)
    for begin in range(0,len(present),config['cache_batch_size']):
        ids=present[begin:begin+config['cache_batch_size']]
        image=torch.stack([original.preprocess(crop(rows[i])) for i in ids]).to(device)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
            value=v._embeds(image)
            for block in v.transformer.resblocks[:-1]:
                value=block(value,attn_mask=None)
            assert value.shape==(len(ids),257,1024) and torch.isfinite(value).all()
            assert value.dtype==(torch.bfloat16 if device=='cuda' else torch.float32)
            dtype=str(value.dtype)
            final=v.transformer.resblocks[-1](value,attn_mask=None)
            pooled,_=v._pool(final)
            reference=F.normalize(pooled @ v.proj,dim=-1).float()
        output[ids]=value.float().cpu().numpy()
        if baseline is not None:
            maximum=max(maximum,float(np.max(np.abs(reference.cpu().numpy()-baseline['features'][ids]))))
        if begin%128==0:
            print('bio prefix',split,begin,'/',len(present),'old feature delta',maximum,flush=True)
    output.flush()
    del output
    pending.replace(destination)
    assert maximum<.02,(split,maximum)
    info={'rows_sha256':sha(rows_path),'rows_path':str(rows_path),'tokens_sha256':sha(destination),
          'prefix_binding_sha256':binding_sha,'images':len(rows),'nonempty_crops':len(present),
          'tokens':[257,1024],'dtype':'float32','compute_token_dtype':dtype,
          'reconstruction_max_abs_error':maximum,'original_features_compared':baseline is not None}
    if test_selection is not None:
        info['selection_sha256']=test_selection
    save(metadata,info)
    return info


def prepare(device='cuda'):
    config=read(CONFIG)
    FOLDER.mkdir(exist_ok=True)
    splits=config['train_splits']+config['validation']
    estimate=sum(len(read(OLD/'features'/(s+'.json'))) for s in splits)*257*1024*4
    assert shutil.disk_usage(HERE).free>estimate+4*1024**3,'Insufficient free space for Bio prefix cache and checkpoints'
    binding={'architecture':'bioclip_tail','config_sha256':sha(CONFIG),**source_binding(),
             'source_sha256':{'bio_tail.py':sha(__file__),'common.py':sha(HERE/'common.py'),
                              'fusion.py':sha(OLD/'fusion.py')},
             'inputs':{str(OLD/'features'/name):sha(OLD/'features'/name) for s in splits
                       for name in (s+'.json',s+'_bio.npz',s+'_dino.npz')},
             'warmstarts':{str(p):sha(p) for seed in config['seeds'] for p,_ in [warmstart(seed)]},
             'prefix':'_embeds, resblocks0..22, before resblock23 and _pool/proj',
             'numeric_device':device,'cache_batch_size':config['cache_batch_size']}
    path=FOLDER/'binding.json'
    if path.exists():
        assert read(path)==binding
    else:
        save(path,binding)
    original=encoder(device)
    assert_visual(original.model.visual)
    before=state_sha(original.model)
    summaries={s:_prepare_one(s,OLD/'features'/(s+'.json'),original,sha(path),device) for s in splits}
    assert state_sha(original.model)==before
    original.close()
    save(FOLDER/'manifest.json',{'architecture':'bioclip_tail','binding_sha256':sha(path),
         **source_binding(),'unchanged_original_visual_sha256':before,'splits':splits,
         'compute_device':device,'no_augmentation':True,'test_images_encoded':False,
         'reconstruction_max_abs_errors':{s:r['reconstruction_max_abs_error'] for s,r in summaries.items()}})


def prepare_test(device='cuda'):
    selection=HERE/'selection.json'
    assert selection.exists(),'Freeze all model and hyperparameter choices before test encoding'
    manifest=read(FOLDER/'manifest.json')
    assert manifest['binding_sha256']==sha(FOLDER/'binding.json')
    rows_path=HERE/'features/test.json'
    assert rows_path.exists() and (HERE/'features/test_dino.npz').exists()
    original=encoder(device)
    assert_visual(original.model.visual)
    before=state_sha(original.model)
    info=_prepare_one('test',rows_path,original,manifest['binding_sha256'],device,sha(selection))
    assert state_sha(original.model)==before
    original.close()
    return info


def _parts(net):
    return {'tail':state_sha(net.block),'norm':state_sha(net.ln_post),
            'proj':mapping_sha({'proj':net.proj}),'head':state_sha(net.head)}


def _optimizer(net,spec,config):
    groups=[{'params':list(net.head.parameters()),'lr':config['head_learning_rate']}]
    if spec['train_tail']:
        groups.append({'params':list(net.block.parameters())+list(net.ln_post.parameters())+[net.proj],
                       'lr':spec['tail_learning_rate']})
    opt=torch.optim.AdamW(groups,weight_decay=config['weight_decay'])
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in net.parameters() if p.requires_grad}
    return opt


def _batch(data,index,device):
    rows,tokens,dino,_=data
    t=torch.from_numpy(np.array(tokens[index],copy=True)).to(device)
    d=torch.from_numpy(np.array(dino[index],copy=True)).to(device)
    y=torch.tensor([int(np.searchsorted(CLASSES,rows[i]['target'])) for i in index],device=device)
    return t,d,y


def _validate(net,validation,device,batch_size):
    net.eval()
    result={}
    with torch.no_grad():
        for split,data in validation.items():
            correct,loss=0,0.
            indexes=data[3]
            for begin in range(0,len(indexes),batch_size):
                t,d,y=_batch(data,indexes[begin:begin+batch_size],device)
                logits=net(t,d)
                correct+=int((logits.argmax(1)==y).sum())
                loss+=float(F.cross_entropy(logits,y,reduction='sum'))
            result[split]={'correct':correct,'images':len(indexes),'loss':loss/len(indexes)}
    return result


def smoke(device='cuda'):
    from smoke import difference, grad_check
    started=time.perf_counter()
    config=read(CONFIG)
    folder=HERE/'bio_smoke'
    folder.mkdir(exist_ok=True)
    seed=config['seeds'][0]
    data=cache('train')
    rows,tokens,dino,supported=data
    ids=supported[:config['cache_batch_size']]
    assert np.array_equal(ids,np.arange(len(ids)))
    t,d,y=_batch(data,ids,device)
    original=encoder(device)
    v=original.model.visual
    assert_visual(v)
    original_state=state_sha(original.model)
    assert original_state==read(FOLDER/'manifest.json')['unchanged_original_visual_sha256']
    batch=torch.stack([original.preprocess(crop(rows[i])) for i in ids]).to(device)
    net=build(seed,True,device).eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
        full=original.model.encode_image(batch,normalize=True).float()
    with torch.no_grad():
        split=net.visual(t)
        actual=net(t,d)
        expected=net.head(full,d)
    numeric=difference(split,full)
    logits_difference=difference(actual,expected)
    assert numeric['max_abs_error']<1e-6,numeric
    assert logits_difference['max_abs_error']<1e-5,logits_difference
    old=load_npz(OLD/'features/train_bio.npz')['features'][ids]
    old_difference=difference(split,torch.from_numpy(np.array(old,copy=True)).to(device))
    assert old_difference['max_abs_error']<.02
    assert all(p.grad is None for p in original.model.parameters())
    assert state_sha(original.model)==original_state
    del v,batch
    original.close()
    # Measure the configured tail-training batch after releasing the frozen full tower.
    t,d,y=_batch(data,supported[:config['batch_size']],device)
    gradient_crops=len(y)
    if device=='cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    before=_parts(net)
    opt=_optimizer(net,next(s for s in config['experiments'] if s['train_tail']),config)
    net.train()
    loss=F.cross_entropy(net(t,d),y,label_smoothing=config['label_smoothing'])
    assert torch.isfinite(loss)
    loss.backward()
    gradients={'tail':grad_check(net.block,True),'ln_post':grad_check(net.ln_post,True),
               'head':grad_check(net.head,True)}
    assert net.proj.grad is not None and torch.isfinite(net.proj.grad).all() and net.proj.grad.float().norm()>0
    gradients['proj_norm']=float(net.proj.grad.float().norm())
    torch.nn.utils.clip_grad_norm_(net.parameters(),config['max_gradient_norm'])
    opt.step()
    after=_parts(net)
    assert all(before[k]!=after[k] for k in before)
    checkpoint=folder/'tail_after_one_step.pt'
    atomic_save(checkpoint,{'model':net.state_dict(),'seed':seed,'train_tail':True,'architecture':'bioclip_tail',
          'global_ids':CLASSES.tolist(),'source_checkpoint_sha256':source_binding()['source_checkpoint_sha256']})
    net.eval()
    with torch.no_grad():
        prediction=net(t[:2],d[:2])
    reloaded,_=restore(checkpoint,device)
    with torch.no_grad():
        assert torch.equal(prediction,reloaded(t[:2],d[:2]))
    control=build(seed,False,device)
    control_before=_parts(control)
    control_opt=_optimizer(control,config['experiments'][0],config)
    control.train()
    control_opt.zero_grad(set_to_none=True)
    control_loss=F.cross_entropy(control(t,d),y,label_smoothing=config['label_smoothing'])
    control_loss.backward()
    grad_check(control.block,False)
    grad_check(control.ln_post,False)
    grad_check(control.head,True)
    assert control.proj.grad is None
    control_opt.step()
    control_after=_parts(control)
    assert all(control_before[k]==control_after[k] for k in ('tail','norm','proj'))
    assert control_before['head']!=control_after['head']
    assert source_binding()['source_checkpoint_sha256']==sha(source_binding()['source_checkpoint'])
    result={'passed':True,'architecture':'bioclip_tail','device':device,'seed':seed,
        'numeric_same_batch':numeric,'head_numeric_same_batch':logits_difference,
        'original_cached_features_difference':old_difference,'loss':float(loss.detach()),'gradients':gradients,
        'tail_before':before,'tail_after':after,'control_before':control_before,'control_after':control_after,
        'class_indices_verified':True,'global_ids':CLASSES.tolist(),'gradient_training_crops':gradient_crops,
        'checkpoint_roundtrip_bitwise_equal':True,'control_tail_ln_post_proj_unchanged':True,
        'original_encoder_unchanged_sha256':original_state,'source_checkpoint_sha256':source_binding()['source_checkpoint_sha256'],
        'trainable_parameters':sum(p.numel() for p in net.parameters() if p.requires_grad),
        'config_sha256':sha(CONFIG),'prefix_manifest_sha256':sha(FOLDER/'manifest.json'),
        'source_sha256':{name:sha(HERE/name) for name in ('bio_tail.py','common.py')},
        'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else None,
        'test_predictions_read':False,'seconds':time.perf_counter()-started}
    save(HERE/'bio_smoke_result.json',result)
    print('Bio smoke passed',numeric,'parameters',result['trainable_parameters'],flush=True)
    return result


def fit(spec,seed,config,device,resume=False):
    output=HERE/'runs'/f"{spec['id']}_{seed}"
    output.mkdir(parents=True,exist_ok=True)
    warm_path,_=warmstart(seed)
    contract={'architecture':'bioclip_tail','spec':spec,'seed':seed,'config_sha256':sha(CONFIG),
        'warmstart_path':str(warm_path),'warmstart_sha256':sha(warm_path),'global_ids':CLASSES.tolist(),
        'prefix_manifest_sha256':sha(FOLDER/'manifest.json'),'data_protocol_sha256':sha(HERE/'data_review/protocol.json'),
        'source_checkpoint_sha256':source_binding()['source_checkpoint_sha256'],
        'source_sha256':{'bio_tail.py':sha(__file__),'common.py':sha(HERE/'common.py')},
        'prefix_metadata_sha256':{s:sha(FOLDER/(s+'.json')) for s in config['train_splits']+config['validation']}}
    if (output/'contract.json').exists():
        assert read(output/'contract.json')==contract
    else:
        save(output/'contract.json',contract)
    if (output/'complete.json').exists():
        result=read(output/'complete.json')
        assert sha(result['checkpoint'])==result['checkpoint_sha256']
        return result
    train=[cache(s) for s in config['train_splits']]
    validation={s:cache(s) for s in config['validation']}
    positions=[(p,int(i)) for p,data in enumerate(train) for i in data[3]]
    counts=np.bincount([int(np.searchsorted(CLASSES,train[p][0][i]['target'])) for p,i in positions],minlength=len(CLASSES))
    assert (counts>0).all()
    weights=torch.tensor(counts,dtype=torch.float32,device=device).rsqrt()
    weights/=weights.mean()
    torch.manual_seed(seed)
    generator=torch.Generator().manual_seed(seed)
    net=build(seed,spec['train_tail'],device)
    before=_parts(net)
    opt=_optimizer(net,spec,config)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,config['epochs'],eta_min=0)
    history,best=[],(-1,-math.inf)
    def checkpoint(epoch):
        return {'model':net.state_dict(),'seed':seed,'train_tail':spec['train_tail'],'epoch':epoch,
                'architecture':'bioclip_tail','global_ids':CLASSES.tolist(),
                'source_checkpoint_sha256':source_binding()['source_checkpoint_sha256'],
                'contract_sha256':sha(output/'contract.json')}
    if (output/'last.pt').exists():
        if not resume:
            raise RuntimeError('Incomplete Bio run; use --resume')
        saved=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
        assert saved['contract_sha256']==sha(output/'contract.json')
        net.load_state_dict(saved['model'],strict=True)
        opt.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        generator.set_state(saved['shuffle_rng'])
        torch.set_rng_state(saved['cpu_rng'])
        if device=='cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'])
        history,best=saved['history'],tuple(saved['best'])
    if not history:
        initial=_validate(net,validation,device,config['batch_size'])
        save(output/'initial_validation.json',initial)
        best=(sum(v['correct'] for v in initial.values()),-sum(v['loss']*v['images'] for v in initial.values()))
        atomic_save(output/'best.pt',checkpoint(0))
    if device=='cuda':
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(len(history)+1,config['epochs']+1):
        net.train()
        total=0.
        order=torch.randperm(len(positions),generator=generator).tolist()
        for begin in range(0,len(order),config['batch_size']):
            chosen=[positions[k] for k in order[begin:begin+config['batch_size']]]
            t=torch.stack([torch.from_numpy(np.array(train[p][1][i],copy=True)) for p,i in chosen]).to(device)
            d=torch.stack([torch.from_numpy(np.array(train[p][2][i],copy=True)) for p,i in chosen]).to(device)
            y=torch.tensor([int(np.searchsorted(CLASSES,train[p][0][i]['target'])) for p,i in chosen],device=device)
            loss=F.cross_entropy(net(t,d),y,weight=weights,label_smoothing=config['label_smoothing'])
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],config['max_gradient_norm'])
            opt.step()
            total+=float(loss.detach())*len(chosen)
        metrics=_validate(net,validation,device,config['batch_size'])
        score=(sum(v['correct'] for v in metrics.values()),-sum(v['loss']*v['images'] for v in metrics.values()))
        history.append({'epoch':epoch,'train_loss':total/len(positions),'validation_correct':score[0],
            'validation_loss':-score[1]/sum(v['images'] for v in metrics.values()),
            **{s:v['correct'] for s,v in metrics.items()},'head_lr':opt.param_groups[0]['lr'],
            'tail_lr':opt.param_groups[-1]['lr'] if spec['train_tail'] else 0.})
        if score>best:
            best=score
            atomic_save(output/'best.pt',checkpoint(epoch))
        scheduler.step()
        atomic_save(output/'last.pt',{**checkpoint(epoch),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
            'history':history,'best':best,'shuffle_rng':generator.get_state(),'cpu_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state() if device=='cuda' else None})
        with (output/'results.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(output.name,'epoch',epoch,'loss',round(total/len(positions),5),'validation',score[0],flush=True)
    after=_parts(net)
    assert all((before[k]!=after[k])==spec['train_tail'] for k in ('tail','norm','proj'))
    selected=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(selected['model'],strict=True)
    chosen=_parts(net)
    result={'architecture':'bioclip_tail','experiment':spec['id'],'seed':seed,'train_tail':spec['train_tail'],
        'train_images':len(positions),'validation_images_with_supported_crops':sum(len(d[3]) for d in validation.values()),
        'best_correct':best[0],'checkpoint':str(output/'best.pt'),'checkpoint_sha256':sha(output/'best.pt'),
        'selected_epoch':selected['epoch'],'tail_before_sha256':before['tail'],'tail_after_sha256':after['tail'],
        'norm_before_sha256':before['norm'],'norm_after_sha256':after['norm'],
        'proj_before_sha256':before['proj'],'proj_after_sha256':after['proj'],
        'selected_tail_sha256':chosen['tail'],'selected_norm_sha256':chosen['norm'],'selected_proj_sha256':chosen['proj'],
        'selected_tail_updated':before['tail']!=chosen['tail'],'selected_norm_updated':before['norm']!=chosen['norm'],
        'selected_proj_updated':before['proj']!=chosen['proj'],
        'selected_visual_backbone_updated':any(before[k]!=chosen[k] for k in ('tail','norm','proj')),
        'head_updated':before['head']!=after['head'],'selected_head_updated':before['head']!=chosen['head'],
        'trainable_parameters':sum(p.numel() for p in net.parameters() if p.requires_grad),
        'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else None}
    save(output/'complete.json',result)
    return result


def train(device='cuda',resume=False):
    assert not (HERE/'selection.json').exists(),'Selection already frozen'
    tested=read(HERE/'bio_smoke_result.json')
    assert tested['passed'] and tested['config_sha256']==sha(CONFIG)
    assert tested['prefix_manifest_sha256']==sha(FOLDER/'manifest.json')
    for name,digest in tested['source_sha256'].items():
        assert sha(HERE/name)==digest
    assert read(HERE/'data_review/protocol.json')['training_can_start']
    config=read(CONFIG)
    binding=read(FOLDER/'binding.json')
    assert binding['config_sha256']==sha(CONFIG)
    for path,digest in {**binding['inputs'],**binding['warmstarts']}.items():
        assert sha(path)==digest
    for split in config['train_splits']+config['validation']:
        assert sha(FOLDER/(split+'.npy'))==read(FOLDER/(split+'.json'))['tokens_sha256']
    for spec in config['experiments']:
        for seed in config['seeds']:
            fit(spec,seed,config,device,resume)
    save(HERE/'bio_training_complete.json',{'architecture':'bioclip_tail','experiments':3,'seeds':config['seeds'],
        'actual_backbone_updated':True,'DINO_backbone_updated':False,'test_used_for_training_or_selection':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','smoke','train','prepare_test'))
    parser.add_argument('--device',default='cuda',choices=('cuda','cpu'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    if args.stage=='smoke':
        save(HERE/'bio_smoke_result.json',{'passed':False,'status':'running'})
        try:
            smoke(args.device)
        except Exception as error:
            save(HERE/'bio_smoke_result.json',{'passed':False,'error':repr(error),'traceback':traceback.format_exc()})
            raise
    elif args.stage=='train':
        train(args.device,args.resume)
    else:
        {'prepare':prepare,'prepare_test':prepare_test}[args.stage](args.device)


if __name__=='__main__':
    main()
