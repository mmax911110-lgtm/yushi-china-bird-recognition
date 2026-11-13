import numpy as np
from scipy.special import softmax,expit
from .classifier import SpeciesHead

class ModelE:
    def __init__(self,root,configuration):
        self.old_text=np.load(root/'models/species/original_text.npz',allow_pickle=False)['features']
        self.negative_text=np.load(root/'models/species/negative_text.npz',allow_pickle=False)['features']
        self.adapted=SpeciesHead.load(root/'models/species/adapted_head.npz')
        self.supported=self.adapted.ids
        self.mixture=configuration['mixture'];self.protect=configuration['protect'];self.offset=configuration['offset']
        self.original_temperature=configuration['original_temperature']
        self.calibration={'temperature':configuration['temperature']};self.gate=configuration['gate']
    def logits(self,x):
        old=x@self.old_text.T*100.;new=self.adapted.scores(x);new[:,self.supported]-=self.offset
        probability=softmax(old/self.original_temperature,axis=1).max(1)
        protect=(~np.isin(old.argmax(1),self.supported))&(probability>=self.protect)
        values=(1-self.mixture)*old+self.mixture*new;values[protect]=old[protect]
        return values,old,~protect
    def gate_score(self,x):
        cosine=x@self.old_text.T;maximum=cosine.max(1);negative=(x@self.negative_text.T).max(1)
        sorted_cos=np.partition(cosine,-2,axis=1)[:,-2:]
        values=np.column_stack([maximum,negative,maximum-negative,sorted_cos.max(1)-sorted_cos.min(1)])
        if self.gate['kind']!='logistic_birdness':raise ValueError('不支持的冻结确认规则')
        p=self.gate['parameters']
        return expit(((values-np.asarray(p['mean']))/np.asarray(p['scale']))@np.asarray(p['coef'])+p['intercept'])
    def predict_features(self,x):
        logits,old,use=self.logits(x);prob=softmax(logits/self.calibration['temperature'],axis=1)
        confidence=prob.max(1);score=self.gate_score(x)
        accepted=(confidence>=self.gate['species_probability_threshold'])&(score>=self.gate['threshold'])
        return {'logits':logits,'confidence':confidence,'gate_score':score,'accepted':accepted,'used_adaptation':use}
