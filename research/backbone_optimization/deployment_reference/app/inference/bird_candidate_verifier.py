from pathlib import Path
import math
import cv2
import numpy as np

PATCH_SIZE=160
CONTEXT=2.0
MIN_SIDE=64


def crop_bounds(box,context=CONTEXT,min_side=MIN_SIDE):
    x1,y1,x2,y2=map(float,box)
    if x2<=x1 or y2<=y1:
        raise ValueError(box)
    side=max(min_side,int(math.ceil(max(x2-x1,y2-y1)*context)))
    left=int(math.floor((x1+x2-side)/2))
    top=int(math.floor((y1+y2-side)/2))
    return left,top,left+side,top+side


def context_patch(frame,box,size=PATCH_SIZE):
    left,top,right,bottom=crop_bounds(box)
    h,w=frame.shape[:2]
    xa,ya,xb,yb=max(0,left),max(0,top),min(w,right),min(h,bottom)
    if xb<=xa or yb<=ya:
        raise ValueError((frame.shape,box))
    crop=frame[ya:yb,xa:xb]
    crop=cv2.copyMakeBorder(crop,ya-top,bottom-yb,xa-left,right-xb,cv2.BORDER_CONSTANT,value=(114,114,114))
    assert crop.shape[0]==crop.shape[1]
    return cv2.resize(crop,(size,size),interpolation=cv2.INTER_LINEAR)


def image_read(path):
    result=cv2.imdecode(np.frombuffer(Path(path).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    if result is None:raise ValueError(path)
    return result


def score_patches(model,patches,device='cuda:0',batch=128):
    import torch
    if not patches:return []
    bird_index=next(int(i) for i,name in model.names.items() if name=='bird')
    output=[]
    net=model.model.to(device).eval().float()
    with torch.inference_mode():
        for start in range(0,len(patches),batch):
            rgb=np.stack([cv2.cvtColor(p,cv2.COLOR_BGR2RGB) for p in patches[start:start+batch]])
            tensor=torch.from_numpy(rgb).permute(0,3,1,2).to(device).float().div_(255)
            result=net(tensor)
            probabilities=result[0] if isinstance(result,tuple) else result
            assert probabilities.ndim==2 and probabilities.shape[1]==2
            assert torch.allclose(probabilities.sum(1),torch.ones(len(probabilities),device=device),atol=.0001)
            output.extend(probabilities[:,bird_index].cpu().tolist())
    return output


def keep_candidates(detector_scores,bird_scores,threshold,bypass):
    return np.asarray([(d>=bypass or b>=threshold) for d,b in zip(detector_scores,bird_scores)],dtype=bool)
