from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
from copy import deepcopy
from pathlib import Path
import json,math
import numpy as np
from scipy.special import softmax,logsumexp
from scipy.optimize import minimize_scalar

VARIANTS=('baseline','P_named_zero','P_taxonomy_zero','P_taxonomy_ridge',
          'P_taxonomy_prototype','blend50_N_named','blend50_N_ridge','blend25_N_ridge')
HEADS=('P_named_zero','P_taxonomy_zero','P_taxonomy_ridge','P_taxonomy_prototype')
BLENDS={'blend50_N_named':('P_named_zero',.5),'blend50_N_ridge':('P_taxonomy_ridge',.5),
        'blend25_N_ridge':('P_taxonomy_ridge',.25)}


def normalize(x):
    x=np.asarray(x,np.float32)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Finite rank2 features required')
    norms=np.linalg.norm(x,axis=1,keepdims=True)
    if len(x) and np.any(norms<1e-9):raise ValueError('Zero embeddings cannot be classified')
    return x/np.maximum(norms,1e-9)


class PHeads:
    def __init__(self,named_text,taxonomy_text,ids,ridge_coef,ridge_intercept,prototypes,
                 ridge_weight=.2,prototype_weight=.2):
        self.named_text=normalize(named_text);self.taxonomy_text=normalize(taxonomy_text)
        if self.named_text.shape!=self.taxonomy_text.shape:raise ValueError('Text matrices differ')
        self.ids=np.asarray(ids,np.int64);self.ridge_coef=np.asarray(ridge_coef,np.float32)
        self.ridge_intercept=np.asarray(ridge_intercept,np.float32);self.prototypes=normalize(prototypes)
        c,d=self.named_text.shape
        if len(set(self.ids.tolist()))!=len(self.ids) or np.any(self.ids<0) or np.any(self.ids>=c):raise ValueError('Invalid supported IDs')
        if self.ridge_coef.shape!=(len(self.ids),d) or self.ridge_intercept.shape!=(len(self.ids),) or self.prototypes.shape!=(len(self.ids),d):raise ValueError('Head dimensions differ')
        self.ridge_weight=float(ridge_weight);self.prototype_weight=float(prototype_weight)
        self.prototype_matrix=self.taxonomy_text.copy()
        self.prototype_matrix[self.ids]=normalize((1-self.prototype_weight)*self.taxonomy_text[self.ids]+self.prototype_weight*self.prototypes)

    @classmethod
    def fit(cls,features,targets,named_text,taxonomy_text,alpha=.1,ridge_weight=.2,prototype_weight=.2):
        from sklearn.linear_model import Ridge
        x=normalize(features);y=np.asarray(targets,np.int64)
        if y.shape!=(len(x),) or not len(y):raise ValueError('One nonempty target per support image required')
        ids,counts=np.unique(y,return_counts=True)
        if np.any(ids<0) or np.any(ids>=len(named_text)):raise ValueError('Support target out of vocabulary')
        count_map=dict(zip(ids,counts));weights=np.asarray([1/count_map[int(i)] for i in y]);weights/=weights.mean()
        targets80=(y[:,None]==ids[None,:]).astype(np.float32)
        estimator=Ridge(alpha=float(alpha),fit_intercept=True,solver='cholesky').fit(x,targets80,sample_weight=weights)
        prototypes=normalize(np.stack([x[y==i].mean(0) for i in ids]))
        result=cls(named_text,taxonomy_text,ids,estimator.coef_,estimator.intercept_,prototypes,ridge_weight,prototype_weight)
        fit_error=estimator.predict(x)-targets80
        stats={'support_images':len(x),'supported_classes':len(ids),'alpha':float(alpha),
               'ridge_target':'one_hot_supported_classes','ridge_contribution':'taxonomy cosine + weight * fitted onehot score on supported IDs; then *100',
               'class_counts':{str(int(i)):int(n) for i,n in zip(ids,counts)},
               'sample_weight_mean':float(weights.mean()),'sample_weight_min':float(weights.min()),'sample_weight_max':float(weights.max()),
               'support_weighted_mean_squared_error':float(np.average((fit_error**2).mean(1),weights=weights)),
               'optimization':'closed-form regularized linear least squares; not neural epochs',
               'backbone_trained':False,'GPU_used':False}
        return result,stats

    def logits(self,features,kind):
        x=normalize(features)
        if x.shape[1]!=self.named_text.shape[1]:raise ValueError('Image/text embedding dimensions differ')
        if kind=='P_named_zero':return 100.*(x@self.named_text.T)
        if kind=='P_taxonomy_zero':return 100.*(x@self.taxonomy_text.T)
        if kind=='P_taxonomy_prototype':return 100.*(x@self.prototype_matrix.T)
        if kind=='P_taxonomy_ridge':
            values=x@self.taxonomy_text.T
            values[:,self.ids]+=self.ridge_weight*(x@self.ridge_coef.T+self.ridge_intercept)
            return values*100.
        raise ValueError('Unknown standalone head: '+kind)

    def probabilities(self,features,baseline_probabilities,temperatures,variants=None):
        requested=tuple(VARIANTS if variants is None else dict.fromkeys(('baseline',*variants)))
        if any(v not in VARIANTS for v in requested):raise ValueError('Unknown candidate')
        base=np.asarray(baseline_probabilities,np.float64)
        if base.ndim!=2 or base.shape!=(len(features),len(self.named_text)):raise ValueError('Baseline probability shape differs')
        if not np.isfinite(base).all() or np.any(base<0) or not np.allclose(base.sum(1),1.,atol=1e-5):raise ValueError('Invalid baseline probabilities')
        needed={BLENDS[v][0] if v in BLENDS else v for v in requested if v!='baseline'}
        head_prob={}
        for name in HEADS:
            if name in needed:
                t=float(temperatures[name])
                if not math.isfinite(t) or t<=0:raise ValueError('Invalid temperature')
                head_prob[name]=softmax(self.logits(features,name).astype(np.float64)/t,axis=1)
        result={}
        for name in requested:
            if name=='baseline':result[name]=base.copy()
            elif name in HEADS:result[name]=head_prob[name]
            else:
                dependency,weight=BLENDS[name];result[name]=(1-weight)*base+weight*head_prob[dependency]
        return result

    def save(self,path):
        with Path(path).open('xb') as f:
            np.savez_compressed(f,named_text=self.named_text,taxonomy_text=self.taxonomy_text,ids=self.ids,
                ridge_coef=self.ridge_coef,ridge_intercept=self.ridge_intercept,prototypes=self.prototypes,
                ridge_weight=np.array(self.ridge_weight),prototype_weight=np.array(self.prototype_weight))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            return cls(z['named_text'],z['taxonomy_text'],z['ids'],z['ridge_coef'],z['ridge_intercept'],z['prototypes'],
                       float(z['ridge_weight']),float(z['prototype_weight']))


def fit_temperature(logits,targets,bounds=(-3.,3.)):
    logits=np.asarray(logits,np.float64);targets=np.asarray(targets,np.int64)
    if logits.ndim!=2 or targets.shape!=(len(logits),) or not len(targets):raise ValueError('Empty or inconsistent calibration')
    if not np.isfinite(logits).all() or np.any(targets<0) or np.any(targets>=logits.shape[1]):raise ValueError('Bad calibration values')
    def objective(logt):
        scaled=logits/np.exp(logt)
        return float(np.mean(logsumexp(scaled,axis=1)-scaled[np.arange(len(targets)),targets]))
    result=minimize_scalar(objective,bounds=tuple(bounds),method='bounded',options={'xatol':1e-6,'maxiter':500})
    if not result.success:raise RuntimeError('Temperature optimization failed')
    t=float(np.exp(result.x))
    if not np.array_equal(logits.argmax(1),(logits/t).argmax(1)):raise AssertionError('Temperature changed Top1')
    return {'temperature':t,'NLL_before':objective(0.),'NLL_after':objective(result.x),
            'images':len(targets),'logT_bounds':list(bounds),'Top1_unchanged':True,'fit_success':True}


def species_rows(probability,labels,variant,original_infos):
    values=np.asarray(probability)
    if values.shape!=(len(original_infos),len(labels)):raise ValueError('Class/box order mismatch')
    if variant=='baseline':return deepcopy(original_infos)
    out=[]
    for i,p in enumerate(values):
        row=deepcopy(original_infos[i]);old_name=row['name'];order=np.argsort(-p,kind='stable')[:5];label=labels[int(order[0])]
        accepted=bool(row['accepted'])
        row.update(name=label['scientific_name'],scientific_name=label['scientific_name'],
            common_name=label.get('common_name') or label['scientific_name'],
            display_name=(label.get('common_name') or label['scientific_name']) if accepted else '鸟（种类不确定）',
            confidence=float(p[order[0]]),classifier='P_bioclip25_'+variant,
            original_N_name=old_name,accepted=accepted,frozen_bio_accepted=accepted,
            confidence_is_validation_temperature_scaled=True,acceptance_from_original_E=True,
            top5=[{'name':labels[int(j)]['scientific_name'],'confidence':float(p[j])} for j in order])
        if row['birdness_score']!=original_infos[i]['birdness_score']:raise AssertionError('Original gate was changed')
        out.append(row)
    return out
