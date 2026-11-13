import numpy as np
def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

class SpeciesHead:
    def __init__(self, text, supported_ids=None, prototypes=None, ridge_coef=None, ridge_intercept=None, kind='text', weight=0.):
        self.text=normalize(np.asarray(text,np.float32));self.ids=np.asarray(supported_ids if supported_ids is not None else [],np.int64)
        self.prototypes=prototypes;self.ridge_coef=ridge_coef;self.ridge_intercept=ridge_intercept
        self.kind=kind;self.weight=weight
        self.matrix=self.text.copy()
        if kind=='prototype':self.matrix[self.ids]=normalize((1-weight)*self.text[self.ids]+weight*prototypes)
    def scores(self,x):
        x=normalize(np.asarray(x,np.float32));values=x@self.matrix.T
        if self.kind=='ridge':
            residual=x@self.ridge_coef.T+self.ridge_intercept
            values[:,self.ids]+=self.weight*residual
        return values*100.
    def export(self,path):
        values={'text':self.text,'supported_ids':self.ids,'kind':np.asarray(self.kind),'weight':np.asarray(self.weight)}
        if self.prototypes is not None:values['prototypes']=self.prototypes
        if self.ridge_coef is not None:values.update(ridge_coef=self.ridge_coef,ridge_intercept=self.ridge_intercept)
        np.savez(path,**values)
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as values:
            return cls(values['text'],values['supported_ids'],
               values['prototypes'] if 'prototypes' in values else None,
               values['ridge_coef'] if 'ridge_coef' in values else None,
               values['ridge_intercept'] if 'ridge_intercept' in values else None,
               str(values['kind']),float(values['weight']))

def support_arrays(x,y):
    ids=np.unique(y)
    prototypes=normalize(np.stack([x[y==i].mean(axis=0) for i in ids]))
    return ids,prototypes
