"""Recompute existing cached baselines only. No new image forward or selection."""
from collections import Counter
from pathlib import Path
import csv
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import softmax
from audit_project import HERE,WORK,DATA,PACKAGE,read,sha,save

LABELS=read(PACKAGE/'configs/china_labels.json')
GLOBAL={r['scientific_name']:i for i,r in enumerate(LABELS)}


def legacy(stage):
    base=WORK/'adaptive_fusion_20261113'
    binding=read(base/f'cache_binding_{stage}.json')
    assert sha(base/f'probabilities_{stage}.npz')==binding['probabilities_sha256']
    with np.load(base/f'probabilities_{stage}.npz',allow_pickle=False) as z:
        a,b=z['p2'],z['p25']
    rows=read(base/f'rows_{stage}.json')
    source=read(DATA/f'phase_p_bioclip25_20261106/evaluation/{stage}/records.json')
    paths={r['row']['key']:r['row']['path'] for r in source}
    result=[]
    for row in rows:
        result.append({**row,'path':paths[row['key']]})
    return result,a,b


def recent(folder):
    from inference.models_e import ModelE
    from inference.head_p import PHeads
    root=WORK/folder/'fresh_test'
    cfg=read(PACKAGE/'configs/recognizer.json')
    model=ModelE(PACKAGE,cfg['species_e'])
    head=PHeads.load(PACKAGE/'models/species/heads_p.npz')
    second={r['key']:r for r in read(root/'second_records.json')}
    rows,aa,bb=[],[],[]
    offset=0
    for item in read(root/'baseline_records.json'):
        row=item['row'];path=root/'inference'/row['key']
        assert sha(path/'base_features.npz')==item['features_sha256']
        assert sha(path/'new_features.npz')==second[row['key']]['features_sha256']
        with np.load(path/'base_features.npz',allow_pickle=False) as z:x=z['features']
        with np.load(path/'new_features.npz',allow_pickle=False) as z:y=z['features']
        count=len(item['boxes']);index=item['primary_index']
        if count:
            aa.append(softmax(model.logits(x)[0].astype(np.float64)/model.calibration['temperature'],axis=1))
            bb.append(softmax(head.logits(y,'P_taxonomy_ridge').astype(np.float64)/cfg['p_temperatures']['P_taxonomy_ridge'],axis=1))
        rows.append({'key':row['key'],'path':row['path'],'target':GLOBAL[row['scientific_name']],
                     'primary':offset+index if index is not None else None,'count':count,
                     'observer_id':row['user_id'],'primary_accepted':bool(item['boxes'][index]['species_info']['accepted']) if index is not None else False})
        offset+=count
    return rows,np.concatenate(aa),np.concatenate(bb)


def evaluate(name,rows,probabilities):
    truth=np.array([r['target'] for r in rows])
    present=np.array([r['primary'] is not None for r in rows])
    p=np.zeros((len(rows),len(LABELS)),np.float64)
    for i,row in enumerate(rows):
        if present[i]:p[i]=probabilities[row['primary']]
    pred=p.argmax(1);pred[~present]=-1
    order=np.argsort(-p,axis=1,kind='stable')[:,:3]
    correct=(pred==truth)&present
    top3=(order==truth[:,None]).any(1)&present
    classes=np.unique(truth)
    precision,recall,f1,support=precision_recall_fscore_support(truth,pred,labels=classes,zero_division=0)
    folder=HERE/'cached_baseline'/name;folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for i,row in enumerate(rows):
        records.append({'key':row['key'],'path':row['path'],'true_global_id':int(truth[i]),
                        'truth':LABELS[int(truth[i])]['scientific_name'],'prediction_global_id':int(pred[i]),
                        'prediction':LABELS[int(pred[i])]['scientific_name'] if present[i] else None,
                        'top3':[LABELS[int(j)]['scientific_name'] for j in order[i]] if present[i] else [],
                        'correct':bool(correct[i]),'has_candidate_box':bool(present[i]),
                        'probability_row':i,'observer_id':row.get('observer_id')})
    def write_json(file,value):
        with (folder/file).open('x',encoding='utf-8') as stream:
            import json
            json.dump(value,stream,ensure_ascii=False,indent=2)
    write_json('records.json',records)
    write_json('correct_keys.json',[r['key'] for r in records if r['correct']])
    write_json('wrong_keys.json',[r['key'] for r in records if not r['correct']])
    write_json('per_species.json',[{'global_id':int(c),'scientific_name':LABELS[int(c)]['scientific_name'],
                 'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i,c in enumerate(classes)])
    with (folder/'probabilities.npz').open('xb') as stream:
        np.savez_compressed(stream,probabilities=p,present=present,targets=truth,global_class_ids=np.arange(len(LABELS)))
    # Sparse confusion matrix includes an explicit no-detection column.
    cells=Counter(zip(truth.tolist(),pred.tolist()))
    write_json('confusion_sparse.json',{'global_labels':[r['scientific_name'] for r in LABELS],
        'no_box_id':-1,'cells':[{'truth_id':a,'prediction_id':b,'count':n} for (a,b),n in sorted(cells.items())]})
    return {'images':len(rows),'target_species':len(classes),'candidate_species':len(LABELS),
            'Top1':float(correct.mean()),'correct':int(correct.sum()),'Top3':float(top3.mean()),
            'MacroF1_observed_target_classes':float(f1.mean()),'macro_class_definition':'Only species with ground-truth support in this dataset',
            'no_boxes':int((~present).sum()),'Top1_conditional_on_candidate_box':float(correct[present].mean()),
            'GPU_forward_performed':False,'data_already_evaluated_historically':True,
            'probability_missing_rule':'No-box rows have zero vectors and present=False; not a valid species distribution',
            'records_sha256':sha(folder/'records.json'),'probabilities_sha256':sha(folder/'probabilities.npz')}


def main():
    result={}
    for stage in ['n_val','o_val','p_test','previous158','previous129']:
        rows,a,b=recent('nonlinear_fusion_20261113' if stage=='previous158' else 'amu_fusion_20261113') if stage.startswith('previous') else legacy(stage)
        result[stage]={}
        for model,prob in [('BioCLIP2_E',a),('deployed_BioCLIP2_plus25',(a+b)/2.)]:
            result[stage][model]=evaluate(f'{stage}_{model}',rows,prob)
        print('CACHED_AUDIT',stage,{m:(v['correct'],v['Top3'],v['MacroF1_observed_target_classes']) for m,v in result[stage].items()},flush=True)
    save('cached_baseline_summary.json',{'basis':'Read-only recomputation on frozen prior YOLO primary crops; not new inference or a pure GT-crop species benchmark',
        'selection_or_training_performed':False,'outputs':result})


if __name__=='__main__':main()
