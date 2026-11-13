"""Development-only encoder complementarity and a CLI input crop."""
import numpy as np
from data import HERE, CLASSES, crop, read, save
from evaluate import load_stage, cached_logits

stages = ('n_val','o_val')
selection = read(HERE/'runs/selection.json')
heads = {c['mode']:c for c in selection['candidates'] if c['form']=='closed' and c['group']=='expanded'}
joined = []
correct_by_method = {name:[] for name in ('bio','dino','fusion','deployed')}
for stage in stages:
    rows,x,y,base = load_stage(stage)
    use = np.array([r['bbox'] is not None and r['target'] in CLASSES for r in rows])
    target = np.array([r['target'] for r in rows])[use]
    joined.append((x[use],y[use]))
    for mode,item in heads.items():
        logits = np.mean(cached_logits(item['models'],x[use],y[use]),axis=0)
        correct_by_method[mode].extend(CLASSES[logits.argmax(1)]==target)
    correct_by_method['deployed'].extend(CLASSES[base['deployed_BioCLIP2_2_5'][use][:,CLASSES].argmax(1)]==target)
bio,dino = (np.concatenate(v).astype(np.float64) for v in zip(*joined))
bio -= bio.mean(0)
dino -= dino.mean(0)
k = bio@bio.T
l = dino@dino.T
cka = float(np.sum(k*l)/(np.linalg.norm(k)*np.linalg.norm(l)))
a,b,f,p = (np.asarray(correct_by_method[k]) for k in ('bio','dino','fusion','deployed'))
save(HERE/'runs/development_diagnostics.json',{'detected_closed_validation_rows':len(a),
    'linear_CKA':cka,'bio_correct_dino_wrong':int((a&~b).sum()),'dino_correct_bio_wrong':int((b&~a).sum()),
    'both_single_backbones_wrong':int((~a&~b).sum()),'fusion_corrected_vs_bio':int((f&~a).sum()),
    'fusion_regressed_vs_bio':int((~f&a).sum()),'fusion_corrected_vs_deployed':int((f&~p).sum()),
    'fusion_regressed_vs_deployed':int((~f&p).sum()),'used_final_test':False})
rows = read(HERE/'features/n_val.json')
crop(next(row for row in rows if row['bbox'] is not None)).save(HERE/'runs/example_validation_crop.png')
