"""Continue the original species head, optionally updating DINO block11 + norm."""
import argparse
import csv
import math
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,OLD,CLASSES,read,save,sha,cache,state_sha,warmstart,atomic_save
from network import build


def tensor_batch(data,index,device):
    rows,tokens,bio,_=data
    x=torch.from_numpy(np.array(bio[index],copy=True)).to(device)
    t=torch.from_numpy(np.array(tokens[index],copy=True)).to(device)
    y=torch.tensor([int(np.searchsorted(CLASSES,rows[i]['target'])) for i in index],device=device)
    return x,t,y


def validate(model,validation,device,batch_size):
    model.eval()
    metrics={}
    with torch.no_grad():
        for name,data in validation.items():
            indexes=data[3]
            correct,loss=0,0.
            for begin in range(0,len(indexes),batch_size):
                x,t,y=tensor_batch(data,indexes[begin:begin+batch_size],device)
                logits=model(x,t)
                correct+=int((logits.argmax(1)==y).sum())
                loss+=float(F.cross_entropy(logits,y,reduction='sum'))
            metrics[name]={'correct':correct,'images':len(indexes),'loss':loss/len(indexes)}
    return metrics


def fit(spec,seed,config,device,resume):
    output=HERE/'runs'/f"{spec['id']}_{seed}"
    output.mkdir(parents=True,exist_ok=True)
    train_parts=[cache(split) for split in config['train_splits']]
    validation={split:cache(split) for split in config['validation']}
    warm_path,_=warmstart(seed)
    contract={'spec':spec,'seed':seed,'config_sha256':sha(HERE/'config.json'),
       'prefix_manifest_sha256':sha(HERE/'tokens/manifest.json'),'warmstart_path':str(warm_path),
       'warmstart_sha256':sha(warm_path),'global_ids':CLASSES.tolist(),
       'data_protocol_sha256':sha(HERE/'data_review/protocol.json'),
       'prefix_metadata_sha256':{split:sha(HERE/'tokens'/(split+'.json'))
                                for split in config['train_splits']+config['validation']},
       'source_sha256':{name:sha(HERE/name) for name in ('common.py','network.py','train_tail.py')},
       'source_heads_and_backbone_are_frozen_inputs':True}
    if (output/'contract.json').exists():
        assert read(output/'contract.json')==contract
    else:
        save(output/'contract.json',contract)
    if (output/'complete.json').exists():
        finished=read(output/'complete.json')
        assert sha(finished['checkpoint'])==finished['checkpoint_sha256']
        return finished
    torch.manual_seed(seed)
    generator=torch.Generator().manual_seed(seed)
    model=build(seed,spec['train_tail'],device)
    tail_before=state_sha(model.block)
    norm_before=state_sha(model.norm)
    head_before=state_sha(model.head)
    parameters=[{'params':model.head.parameters(),'lr':config['head_learning_rate']}]
    if spec['train_tail']:
        parameters.append({'params':list(model.block.parameters())+list(model.norm.parameters()),
                           'lr':spec['tail_learning_rate']})
    optimizer=torch.optim.AdamW(parameters,weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config['epochs'],eta_min=0)
    positions=[(part,i) for part,data in enumerate(train_parts) for i in data[3]]
    counts=np.bincount([int(np.searchsorted(CLASSES,train_parts[p][0][i]['target'])) for p,i in positions],minlength=len(CLASSES))
    assert (counts>0).all()
    weights=torch.tensor(counts,dtype=torch.float32,device=device).rsqrt()
    weights/=weights.mean()
    history=[]
    best=(-1,-math.inf)
    if (output/'last.pt').exists():
        if not resume:
            raise RuntimeError('Incomplete run: use --resume')
        state=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
        assert state['contract_sha256']==sha(output/'contract.json')
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        generator.set_state(state['shuffle_rng'])
        torch.set_rng_state(state['cpu_rng'])
        if device=='cuda':
            torch.cuda.set_rng_state(state['cuda_rng'])
        history,best=state['history'],tuple(state['best'])
    initial=validate(model,validation,device,config['batch_size']) if not history else None
    if not history:
        save(output/'initial_validation.json',initial)
        best=(sum(v['correct'] for v in initial.values()),-sum(v['loss']*v['images'] for v in initial.values()))
        atomic_save(output/'best.pt',{'model':model.state_dict(),'seed':seed,'train_tail':spec['train_tail'],
             'epoch':0,'global_ids':CLASSES.tolist(),'contract_sha256':sha(output/'contract.json')})
    if device=='cuda':
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(len(history)+1,config['epochs']+1):
        model.train()
        total=0.
        permutation=torch.randperm(len(positions),generator=generator).tolist()
        for begin in range(0,len(permutation),config['batch_size']):
            indexes=permutation[begin:begin+config['batch_size']]
            # Preserve shuffled order across the two fixed training splits.
            x=torch.stack([torch.from_numpy(np.array(train_parts[positions[k][0]][2][positions[k][1]],copy=True)) for k in indexes]).to(device)
            t=torch.stack([torch.from_numpy(np.array(train_parts[positions[k][0]][1][positions[k][1]],copy=True)) for k in indexes]).to(device)
            target=torch.tensor([int(np.searchsorted(CLASSES,train_parts[positions[k][0]][0][positions[k][1]]['target'])) for k in indexes],device=device)
            loss=F.cross_entropy(model(x,t),target,weight=weights,label_smoothing=config['label_smoothing'])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config['max_gradient_norm'])
            optimizer.step()
            total+=float(loss.detach())*len(indexes)
        validation_metrics=validate(model,validation,device,config['batch_size'])
        score=(sum(v['correct'] for v in validation_metrics.values()),-sum(v['loss']*v['images'] for v in validation_metrics.values()))
        history.append({'epoch':epoch,'train_loss':total/len(positions),
            'validation_correct':score[0],'validation_loss':-score[1]/sum(v['images'] for v in validation_metrics.values()),
            **{name:value['correct'] for name,value in validation_metrics.items()},
            'head_lr':optimizer.param_groups[0]['lr'],
            'tail_lr':optimizer.param_groups[-1]['lr'] if spec['train_tail'] else 0.})
        checkpoint={'model':model.state_dict(),'seed':seed,'train_tail':spec['train_tail'],'epoch':epoch,
                    'global_ids':CLASSES.tolist(),'contract_sha256':sha(output/'contract.json')}
        if score>best:
            best=score
            atomic_save(output/'best.pt',checkpoint)
        scheduler.step()
        atomic_save(output/'last.pt',{**checkpoint,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
             'history':history,'best':best,'shuffle_rng':generator.get_state(),'cpu_rng':torch.get_rng_state(),
             'cuda_rng':torch.cuda.get_rng_state() if device=='cuda' else None})
        with (output/'results.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(output.name,'epoch',epoch,'train loss',round(total/len(positions),5),'validation',score[0],flush=True)
    tail_after=state_sha(model.block)
    norm_after=state_sha(model.norm)
    head_after=state_sha(model.head)
    assert (tail_before!=tail_after)==spec['train_tail']
    assert (norm_before!=norm_after)==spec['train_tail']
    selected=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(selected['model'],strict=True)
    selected_tail=state_sha(model.block)
    selected_norm=state_sha(model.norm)
    result={'experiment':spec['id'],'seed':seed,'train_tail':spec['train_tail'],'train_images':len(positions),
        'validation_images_with_supported_crops':sum(len(v[3]) for v in validation.values()),
        'best_correct':best[0],'checkpoint':str(output/'best.pt'),'checkpoint_sha256':sha(output/'best.pt'),
        'tail_before_sha256':tail_before,'tail_after_sha256':tail_after,'norm_before_sha256':norm_before,
        'norm_after_sha256':norm_after,'head_updated':head_before!=head_after,
        'selected_epoch':selected['epoch'],'selected_tail_sha256':selected_tail,
        'selected_norm_sha256':selected_norm,
        'selected_visual_backbone_updated':selected_tail!=tail_before or selected_norm!=norm_before,
        'selected_head_updated':head_before!=state_sha(model.head),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else None}
    save(output/'complete.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda',choices=('cuda','cpu'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if (HERE/'selection.json').exists():
        raise RuntimeError('Experiment already frozen')
    assert read(HERE/'smoke_result.json')['passed']
    assert read(HERE/'data_review/protocol.json')['training_can_start']
    config=read(HERE/'config.json')
    for split in config['train_splits']+config['validation']:
        metadata=read(HERE/'tokens'/(split+'.json'))
        assert sha(HERE/'tokens'/(split+'.npy'))==metadata['tokens_sha256']
    for spec in config['experiments']:
        for seed in config['seeds']:
            fit(spec,seed,config,args.device,args.resume)
    save(HERE/'training_complete.json',{'experiments':len(config['experiments']),'seeds':config['seeds'],
        'actual_backbone_updated':True,'test_used_for_training_or_selection':False})


if __name__=='__main__':
    main()
