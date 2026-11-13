"""Select on development data, then evaluate frozen candidates on the new test."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import precision_recall_fscore_support
import torch
from inference.models_e import ModelE
from inference.head_p import PHeads
from data import HERE, PACKAGE, CLASSES, LABELS, read, save, save_npz, sha
from train import CACHE, load, load_head


def raw_base(x, x25):
    config = read(PACKAGE/'configs/recognizer.json')
    old = ModelE(PACKAGE,config['species_e'])
    p = PHeads.load(PACKAGE/'models/species/heads_p.npz')
    a = softmax(old.logits(x)[0].astype(np.float64)/old.calibration['temperature'],axis=1)
    b = softmax(p.logits(x25,'P_taxonomy_ridge').astype(np.float64)/config['p_temperatures']['P_taxonomy_ridge'],axis=1)
    return {'E0_BioCLIP2':a,'deployed_BioCLIP2_2_5':(a+b)/2.}


def calibration(logits, target):
    result=minimize_scalar(lambda value: float((logsumexp(logits/np.exp(value),axis=1)
        - (logits/np.exp(value))[np.arange(len(target)),target]).mean()),bounds=(-2.3,2.3),method='bounded')
    assert result.success
    return float(np.exp(result.x))


def evaluate(rows, probabilities, classes, include_records=False):
    true=np.array([r['target'] for r in rows])
    present=np.array([r['bbox'] is not None for r in rows])
    accepted=np.array([r['accepted'] for r in rows])
    first = classes[probabilities.argmax(1)].copy()
    first[~present]=-1
    top = classes[np.argsort(-probabilities,axis=1,kind='stable')[:,:3]]
    correct=first==true
    valid_classes=np.unique(true)
    precision,recall,f1,support=precision_recall_fscore_support(true,first,labels=valid_classes,zero_division=0)
    result={'images':len(rows),'correct':int(correct.sum()),'top1':float(correct.mean()),
        'top3_correct':int(((top==true[:,None]).any(1)&present).sum()),'macro_f1':float(f1.mean()),
        'no_box':int((~present).sum()),'accepted_correct':int((correct&accepted).sum()),
        'accepted_wrong':int((~correct&accepted).sum()),'species':len(valid_classes)}
    if include_records:
        result.update({'rows':[{'key':r['key'],'path':r['path'],'target':int(t),'prediction':int(p),
            'has_box':bool(box),'accepted':bool(r['accepted']),'correct':bool(good),'user_id':r['user_id'],
            'top3':[int(i) for i in top_ids] if box else []} for r,t,p,box,good,top_ids in zip(rows,true,first,present,correct,top)],
            'per_species':[{'id':int(c),'name':LABELS[int(c)]['scientific_name'],'precision':float(prec),
            'recall':float(rec),'f1':float(f),'support':int(n)} for c,prec,rec,f,n in zip(valid_classes,precision,recall,f1,support)],
            'confusion':[{'target':int(t),'prediction':int(p),'count':int(n)} for (t,p),n in sorted(Counter(zip(true,first)).items())]})
    return result


def cached_logits(models, x, y):
    outputs=[]
    with torch.inference_mode():
        for item in models:
            assert sha(item['checkpoint'])==item['checkpoint_sha256']
            head=load_head(item['checkpoint'])
            outputs.append(head(torch.from_numpy(x),torch.from_numpy(y)).numpy().astype(np.float64))
    return outputs


def apply_residual(base, logits, alpha, temperature):
    """Centered learned evidence in the known classes, preserving outside winners."""
    eligible=np.isin(base.argmax(1),CLASSES)
    out=base.copy()
    if eligible.any():
        proposal=np.log(np.maximum(base[eligible],1e-300))
        signal=logits[eligible]/temperature
        signal=signal-signal.mean(1,keepdims=True)
        proposal[:,CLASSES] += alpha * signal
        ranked=softmax(proposal,axis=1)
        valid=np.isin(ranked.argmax(1),CLASSES)
        out[np.flatnonzero(eligible)[valid]]=ranked[valid]
    return out


def load_stage(stage):
    rows=read(CACHE/f'{stage}.json')
    x,y,z=(load(stage,kind) for kind in ('bio','dino','bio25'))
    present=np.array([r['bbox'] is not None for r in rows],dtype=bool)
    scored=raw_base(x[present],z[present])
    base={}
    for name,values in scored.items():
        base[name]=np.full((len(rows),len(LABELS)),1./len(LABELS),dtype=np.float64)
        base[name][present]=values
    return rows,x,y,base


def select():
    output=HERE/'runs/selection.json'
    if output.exists():
        raise RuntimeError('Selection already frozen.')
    config=read(HERE/'config.json')
    stages={stage:load_stage(stage) for stage in config['validation']}
    closed={stage:np.array([r['target'] in CLASSES for r in value[0]]) for stage,value in stages.items()}
    base_name='deployed_BioCLIP2_2_5'
    all_ids=np.arange(len(LABELS))
    baselines={name:{s:{'full':evaluate(rows,base[name],all_ids),
                       'closed':evaluate([r for i,r in enumerate(rows) if closed[s][i]],base[name][closed[s]][:,CLASSES],CLASSES)}
            for s,(rows,_,_,base) in stages.items()} for name in ('E0_BioCLIP2',base_name)}
    trials=[]
    candidates=[]
    for group in config['training_groups']:
        for mode in ('bio','dino','fusion'):
            models=[read(HERE/'runs'/f'{group}_{mode}_{seed}'/'complete.json') for seed in config['seeds']]
            per_stage={s:cached_logits(models,x,y) for s,(_,x,y,_) in stages.items()}
            for tag,indexes in [('all',list(range(len(models))))]+[(str(seed),[i]) for i,seed in enumerate(config['seeds'])]:
                model_list=[models[i] for i in indexes]
                averaged={s:np.mean([v[i] for i in indexes],axis=0) for s,v in per_stage.items()}
                fit_logits,fit_targets=[],[]
                for s, (rows,_,_,_) in stages.items():
                    valid=closed[s]&np.array([r['bbox'] is not None for r in rows])
                    fit_logits.append(averaged[s][valid])
                    fit_targets.extend(np.searchsorted(CLASSES,[r['target'] for i,r in enumerate(rows) if valid[i]]))
                temperature=calibration(np.concatenate(fit_logits),np.asarray(fit_targets))
                spec={'id':f'{group}_{mode}_{tag}','mode':mode,'group':group,'seed_tag':tag,'models':model_list,'temperature':temperature}
                metrics={s:evaluate([r for i,r in enumerate(rows) if closed[s][i]],
                    softmax(averaged[s][closed[s]]/temperature,axis=1),CLASSES) for s,(rows,_,_,_) in stages.items()}
                candidates.append({**spec,'form':'closed','metrics':metrics})
                if tag!='all':
                    continue
                for alpha in config['residual_weights']:
                    metrics={s:evaluate(rows,apply_residual(base[base_name],averaged[s],alpha,temperature),all_ids)
                        for s,(rows,_,_,base) in stages.items()}
                    trials.append({**spec,'id':spec['id']+f'_residual{alpha:g}', 'form':'full_residual','alpha':alpha,'metrics':metrics})
    base=sum(v['full']['correct'] for v in baselines[base_name].values())
    score=lambda c:(sum(m['correct'] for m in c['metrics'].values()),sum(m['macro_f1'] for m in c['metrics'].values()))
    # A single primary candidate is chosen before test. The four architectures/data
    # controls also remain fixed for a paired comparison, with no reselection on test.
    finalists=[max([c for c in candidates if c['group']==group and c['mode']==mode and c['seed_tag']=='all'],key=score)
        for group in config['training_groups'] for mode in ('bio','dino','fusion')]
    best_fusion=max([c for c in trials if c['mode']=='fusion'],key=score)
    passing=[c for c in trials if score(c)[0]>base and all(
        c['metrics'][s]['correct']>=baselines[base_name][s]['full']['correct'] for s in stages)]
    selected=max(passing,key=score) if passing else None
    seen={}
    for candidate in [*finalists,best_fusion,selected]:
        if candidate is not None:
            seen[candidate['id']]=candidate
    save(output,{'candidates':list(seen.values()),'deployment_candidate':selected,
        'scope':'one frozen comparison of all named controls; no test-driven reselection',
        'development_baselines':baselines,'all_closed_trials':candidates,'all_residual_trials':trials,
        'validation_model_selection':True,'test_features_read':False,
        'config_sha256':sha(HERE/'config.json'),'frozen_data_sha256':sha(HERE/'frozen_data/manifest.json')})
    print('Closed',[(c['id'],score(c)[0]) for c in finalists],flush=True)
    print('Full baseline',base,'best fusion residual',best_fusion['id'],score(best_fusion)[0],
          'deployment candidate',None if selected is None else selected['id'],flush=True)


def test():
    selection=read(HERE/'runs/selection.json')
    assert sha(HERE/'frozen_data/manifest.json')==selection['frozen_data_sha256']
    assert sha(HERE/'config.json')==selection['config_sha256']
    destination=HERE/'runs/test'
    destination.mkdir(exist_ok=True)
    if (destination/'summary.json').exists():
        raise RuntimeError('Final test already scored.')
    rows,x,y,base=load_stage('test')
    assert all(r['target'] in CLASSES for r in rows)
    full_ids=np.arange(len(LABELS))
    result={}
    predictions={}
    for name,p in base.items():
        predictions[name+'_full']=(p,full_ids)
        closed=p[:,CLASSES]
        predictions[name+'_closed']=(closed/closed.sum(1,keepdims=True),CLASSES)
    for spec in selection['candidates']:
        logits=np.mean(cached_logits(spec['models'],x,y),axis=0)
        if spec['form']=='closed':
            predictions[spec['id']]=(softmax(logits/spec['temperature'],axis=1),CLASSES)
        else:
            predictions[spec['id']]=(apply_residual(base['deployed_BioCLIP2_2_5'],logits,spec['alpha'],spec['temperature']),full_ids)
    baselines={len(classes):evaluate(rows,p,classes,True)['rows'] for p,classes in (
        predictions['deployed_BioCLIP2_2_5_full'],predictions['deployed_BioCLIP2_2_5_closed'])}
    for name,(p,classes) in predictions.items():
        measured=evaluate(rows,p,classes,True)
        detail=measured.pop('rows')
        reference=baselines[len(classes)]
        corrected=[a['key'] for a,b in zip(detail,reference) if a['correct'] and not b['correct']]
        regressed=[a['key'] for a,b in zip(detail,reference) if not a['correct'] and b['correct']]
        present=np.array([r['has_box'] for r in detail])
        save(destination/(name+'.json'),{'records':detail,**measured,'corrected':corrected,'regressed':regressed})
        save_npz(destination/(name+'.npz'),probabilities=np.where(present[:,None],p,0),class_ids=classes,present=present)
        result[name]={k:v for k,v in measured.items() if k not in ('confusion','per_species')}
        result[name].update(corrected=len(corrected),regressed=len(regressed),class_count=len(classes))
        print(name,result[name]['correct'],'/',len(rows),'delta',len(corrected)-len(regressed),flush=True)
    save(destination/'summary.json',{'metrics':result,'selection_sha256':sha(HERE/'runs/selection.json'),
        'test_manifest_sha256':sha(HERE/'frozen_data/test.json'),'test_stage_rows_sha256':sha(CACHE/'test.json'),
        'test_retuning':False,'fixed_detection':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['select','test'])
    arguments=parser.parse_args()
    torch.set_num_threads(4)
    {'select':select,'test':test}[arguments.stage]()
