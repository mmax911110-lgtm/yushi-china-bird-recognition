from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import gc, json, os, sys, time
import numpy as np
from PIL import Image

VARIANTS=('ordinary','low640','rescue1280')

def overlap(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    aa=max(1e-9,(a[2]-a[0])*(a[3]-a[1]));bb=max(1e-9,(b[2]-b[0])*(b[3]-b[1]))
    return inter/max(1e-9,aa+bb-inter),inter/min(aa,bb)

def duplicate(a,b):
    iou,contained=overlap(a,b);return iou>=.5 or contained>=.8


def partial_existing_bird(candidate,existing,window):
    if window is None:return False
    wx1,wy1,wx2,wy2=window;ex1,ey1,ex2,ey2=existing
    if wx1<=ex1 and wy1<=ey1 and ex2<=wx2 and ey2<=wy2:return False
    visible=(max(wx1,ex1),max(wy1,ey1),min(wx2,ex2),min(wy2,ey2))
    vx1,vy1,vx2,vy2=visible
    if vx2<=vx1 or vy2<=vy1:return False
    cx1,cy1,cx2,cy2=candidate;cx,cy=(cx1+cx2)/2,(cy1+cy2)/2
    vx,vy=(vx1+vx2)/2,(vy1+vy2)/2
    return vx1<=cx<=vx2 and vy1<=cy<=vy2 and cx1<=vx<=cx2 and cy1<=vy<=cy2

def square_context(image,xyxy):
    x1,y1,x2,y2=map(float,xyxy)
    if not all(np.isfinite([x1,y1,x2,y2])) or x2<=x1 or y2<=y1:raise ValueError('Invalid crop box')
    side=max(2,int(np.ceil(max(x2-x1,y2-y1)*1.2)))
    left=int(np.floor((x1+x2-side)/2));top=int(np.floor((y1+y2-side)/2))
    bounds=(max(0,left),max(0,top),min(image.width,left+side),min(image.height,top+side))
    if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:raise ValueError('Crop is outside image')
    square=Image.new('RGB',(side,side),(114,114,114));square.paste(image.crop(bounds),(bounds[0]-left,bounds[1]-top))
    return square

class EngineN:
    def __init__(self,device='cuda',*,detector,encoder,species,labels,batch_size=4):
        self.device=device;self.batch_size=batch_size;self.lora_path=None
        self.detector=detector;self.encoder=encoder;self.species=species;self.labels=labels;self.tail=None

    def _classify(self,crops):
        if not crops:return np.empty((0,768),np.float32),np.empty((0,768),np.float32),[]
        base,_=self.encoder.encode_images(crops);adapted=base
        answer=self.species.predict_features(base)
        ranking=answer['logits']
        from scipy.special import softmax
        probabilities=softmax(ranking/self.species.calibration['temperature'],axis=1)
        base_prob=softmax(answer['logits']/self.species.calibration['temperature'],axis=1)
        infos=[]
        for i,p in enumerate(probabilities):
            order=np.argsort(-p)[:5];best=int(order[0]);original=int(base_prob[i].argmax());label=self.labels[best]
            accepted=bool(answer['accepted'][i])
            infos.append({'name':label['scientific_name'],'scientific_name':label['scientific_name'],
                'common_name':label.get('common_name') or label['scientific_name'],
                'display_name':(label.get('common_name') or label['scientific_name']) if accepted else '鸟（种类不确定）',
                'accepted':accepted,'frozen_bio_accepted':accepted,'confidence':float(p[best]),
                'birdness_score':float(answer['gate_score'][i]),'feature_index':i,
                'classifier':'E',
                'base_name':self.labels[original]['scientific_name'],'base_confidence':float(base_prob[i,original]),
                'top5':[{'name':self.labels[int(j)]['scientific_name'],'confidence':float(p[j])} for j in order]})
        return base,adapted,infos

    def predict(self,image,variant='ordinary',initial_detection=None,tile_windows=None):
        if variant not in VARIANTS:raise ValueError(variant)
        image=image.convert('RGB');self.encoder.sync();start=time.perf_counter()
        ordinary=self.detector.detect(image,mode='standard',render=False) if initial_detection is None else initial_detection
        legacy=deepcopy(ordinary.boxes);self.encoder.sync();ordinary_ms=(time.perf_counter()-start)*1000
        if initial_detection is not None:ordinary_ms=0.
        verified=[];raw_count=unmatched_count=0;extra_ms=0.;background_removed=0
        if variant!='ordinary':
            begin=time.perf_counter();frame=np.asarray(image)[:,:,::-1].copy();old_size=self.detector.config['imgsz']
            try:
                self.detector.config['imgsz']=640 if variant=='low640' else 1280
                raw=self.detector._predict(frame,confidence=.05)
                raw_count=len(raw)
                candidates=[r for r in raw if (variant!='low640' or r['confidence']<.16)
                    and not any(duplicate(r['xyxy'],b['xyxy']) for b in legacy)]
                unmatched_count=len(candidates)
                verified,background_removed=self.detector._verify(frame,candidates)
            finally:self.detector.config['imgsz']=old_size
            self.encoder.sync();extra_ms=(time.perf_counter()-begin)*1000
        tile_scan=self.detector.detect_tiles(image,tile_windows) if tile_windows else None
        candidate_boxes=legacy+deepcopy(verified)
        crops=[square_context(image,b['xyxy']) for b in candidate_boxes]
        begin=time.perf_counter();base,adapted,infos=self._classify(crops);self.encoder.sync();classify_ms=(time.perf_counter()-begin)*1000
        all_rows=[]
        for i,(box,info) in enumerate(zip(candidate_boxes,infos)):
            value=deepcopy(box);value['species_info']=info
            all_rows.append({'box':value,'feature_index':i,'origin':'ordinary' if i<len(legacy) else variant,
                'passed_frozen_E_gate':info['accepted'],'retained':i<len(legacy),'drop_reason':None if i<len(legacy) else 'frozen_E_rejected'})
        final=[r['box'] for r in all_rows[:len(legacy)]];added=[]
        for index in sorted(range(len(legacy),len(all_rows)),key=lambda i:-all_rows[i]['box']['confidence']):
            item=all_rows[index];box=item['box']
            if not item['passed_frozen_E_gate']:continue
            if any(duplicate(box['xyxy'],b['xyxy']) for b in final):item['drop_reason']='overlap_duplicate';continue
            item['retained']=True;item['drop_reason']=None;final.append(box);added.append(box)
        for original,current in zip(legacy,final):assert {k:v for k,v in current.items() if k!='species_info'}==original
        assert len(final)>=len(legacy)
        primary_pool=range(len(legacy)) if legacy else range(len(final))
        primary=max(primary_pool,key=lambda i:final[i]['confidence'],default=None)
        self.encoder.sync();elapsed=(time.perf_counter()-start)*1000
        result={'variant':variant,'device':self.device,'legacy_boxes':legacy,'boxes':final,'added':added,
            'primary_index':primary,'legacy_fields_preserved':True,'ordinary_prefix_length':len(legacy),
            'detector_passes':ordinary.passes+(variant!='ordinary'),'tiled':False,'candidate_records':all_rows,
            'ordinary_candidates':ordinary.detector_candidates,'ordinary_filtered':ordinary.filtered_candidates,
            'extra_raw_candidates':raw_count,'extra_unmatched_candidates':unmatched_count,
            'extra_background_verified':len(verified),'extra_background_filtered':background_removed,
            'extra_E_rejected':sum(not r['passed_frozen_E_gate'] for r in all_rows[len(legacy):]),
            'ordinary_detector_ms':ordinary_ms,'extra_detector_ms':extra_ms,'classification_ms':classify_ms,'total_ms':elapsed,
            'lora_checkpoint':self.lora_path,'_base_features':base,'_adapted_features':adapted,'_crops':crops}
        if tile_scan is not None:result['_tile_result']=self._classify_tiles(image,tile_scan,final)
        return result

    def _classify_tiles(self,image,scan,normal_boxes):
        rows=[];candidates=[]
        for candidate in scan['verified']:
            item={'box':deepcopy(candidate),'feature_index':None,'origin':'tile',
                'passed_frozen_E_gate':False,'retained':False,'drop_reason':'normal_overlap'}
            rows.append(item)
            if any(duplicate(candidate['xyxy'],b['xyxy']) for b in normal_boxes):continue
            partial_index=next((i for i,b in enumerate(normal_boxes)
                if partial_existing_bird(candidate['xyxy'],b['xyxy'],candidate.get('tile_window'))),None)
            if partial_index is not None:
                item['drop_reason']='partial_existing_bird';item['duplicate_normal_index']=partial_index
                continue
            item['feature_index']=len(candidates);item['drop_reason']='frozen_E_rejected';candidates.append(item)
        crops=[square_context(image,r['box']['xyxy']) for r in candidates]
        begin=time.perf_counter();base,adapted,infos=self._classify(crops);self.encoder.sync()
        classify_ms=(time.perf_counter()-begin)*1000
        for row,info in zip(candidates,infos):row['box']['species_info']=info;row['passed_frozen_E_gate']=info['accepted']
        final=[]
        for row in sorted(candidates,key=lambda r:-r['box']['confidence']):
            if not row['passed_frozen_E_gate']:continue
            if any(duplicate(row['box']['xyxy'],b['xyxy']) for b in final):row['drop_reason']='tile_overlap';continue
            row['retained']=True;row['drop_reason']=None;final.append(row['box'])
        return {'boxes':final,'candidate_records':rows,'classification_ms':classify_ms,'scan':scan,
            '_base_features':base,'_adapted_features':adapted,'_crops':crops}

    def close(self):
        self.detector=None;self.encoder=None;self.tail=None;self.species=None;gc.collect()
        if self.device=='cuda':
            import torch
            torch.cuda.empty_cache()
