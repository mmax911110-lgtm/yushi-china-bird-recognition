from types import SimpleNamespace
from copy import deepcopy
import time
import numpy as np
from .bird_candidate_verifier import context_patch,score_patches
from .tiling import map_tile_box

class Detector:
    def __init__(self,root,config,device):
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        self.config=deepcopy(config);self.device='0' if device=='cuda' else 'cpu'
        self.verifier_device='cuda:0' if device=='cuda' else 'cpu'
        self.classes=config['classes']
        self.detector=YOLO(str(root/'models/detection/detector_reference.pt'))
        self.classifier=YOLO(str(root/'models/detection/bird_background_verifier.pt'))
        if self.detector.task!='detect' or self.classifier.task!='classify':raise ValueError('模型任务类型不正确')
        self.classifier.model.to(self.verifier_device).eval().float()
    def _predict(self,frame,confidence=None):
        prediction=self.detector.predict(frame,imgsz=self.config.get('imgsz',640),
            conf=self.config['conf'] if confidence is None else confidence,
            iou=self.config.get('iou',.7),classes=self.classes,device=self.device,save=False,verbose=False)[0]
        return [{'xyxy':box,'confidence':float(score),'class_id':int(class_id),
                 'class_name':str(self.detector.names[int(class_id)])}
                for box,score,class_id in zip(prediction.boxes.xyxy.cpu().tolist(),
                    prediction.boxes.conf.cpu().tolist(),prediction.boxes.cls.cpu().tolist())]
    def _verify(self,frame,candidates):
        weak=[i for i,row in enumerate(candidates) if row['confidence']<self.config['detector_bypass']]
        keep=[True]*len(candidates)
        if weak:
            probabilities=score_patches(self.classifier,[context_patch(frame,candidates[i]['xyxy']) for i in weak],device=self.verifier_device)
            for index,probability in zip(weak,probabilities):keep[index]=probability>=self.config['verifier_threshold']
        return [row for row,k in zip(candidates,keep) if k],len(candidates)-sum(keep)
    def detect(self,image,mode='standard',render=False):
        if mode!='standard':raise ValueError('初检必须使用整图检测')
        frame=np.asarray(image)[:,:,::-1].copy();candidates=self._predict(frame);boxes,filtered=self._verify(frame,candidates)
        for row in boxes:row['original_xyxy']=list(row['xyxy'])
        return SimpleNamespace(boxes=boxes,passes=1,detector_candidates=len(candidates),filtered_candidates=filtered)

    def detect_tiles(self,image,windows):
        import torch
        if self.verifier_device!='cpu':torch.cuda.synchronize()
        start=time.perf_counter();frame=np.asarray(image)[:,:,::-1].copy()
        records=[];verified=[];window_records=[];old_size=self.config['imgsz']
        try:
            self.config['imgsz']=640
            for tile_index,window in enumerate(windows):
                left,top,right,bottom=window
                raw=self._predict(frame[top:bottom,left:right].copy(),confidence=.05)
                candidates=[];local_records=[]
                for row in raw:
                    mapped,cut=map_tile_box(row['xyxy'],window,image.size)
                    item={'tile_index':tile_index,'window_xyxy':list(window),'local_xyxy':list(row['xyxy']),
                        'xyxy':mapped,'confidence':row['confidence'],'cut_edges':cut,
                        'status':'internal_boundary_rejected' if cut else 'pending_background'}
                    records.append(item);local_records.append(item)
                    if cut:continue
                    candidate=deepcopy(row);candidate.update(xyxy=mapped,detection_origin='tile',tile_index=tile_index,tile_window=list(window))
                    candidates.append(candidate);item['candidate']=candidate
                kept,filtered=self._verify(frame,candidates)
                kept_ids={id(b) for b in kept}
                for item in local_records:
                    if 'candidate' in item:item['status']='background_passed' if id(item['candidate']) in kept_ids else 'background_rejected'
                verified.extend(kept)
                window_records.append({'tile_index':tile_index,'xyxy':list(window),'raw_candidates':len(raw),
                    'boundary_rejected':sum(bool(r['cut_edges']) for r in local_records),'background_rejected':filtered,'background_passed':len(kept)})
        finally:self.config['imgsz']=old_size
        if self.verifier_device!='cpu':torch.cuda.synchronize()
        return {'windows':window_records,'candidate_records':records,'verified':verified,
            'raw_candidates':len(records),'boundary_rejected':sum(bool(r['cut_edges']) for r in records),
            'background_rejected':sum(r['status']=='background_rejected' for r in records),
            'detector_ms':(time.perf_counter()-start)*1000}
