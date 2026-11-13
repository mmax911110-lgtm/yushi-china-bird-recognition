from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
import gc,hashlib,json,math,os,time,uuid
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['YOLO_OFFLINE']='true'
os.environ['YOLO_AUTOINSTALL']='false'
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
from scipy.special import softmax
from inference.engine_n import EngineN,square_context
from inference.models_e import ModelE
from inference.head_p import PHeads,species_rows
from inference.tiling import tile_windows,make_initial_packet,validate_initial_packet

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
def package_path(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('模型配置不能引用软件目录以外的文件')
    return path
def read_image(path):
    with Image.open(path) as f:return ImageOps.exif_transpose(f).convert('RGB')

def verify_assets(root,config,progress=None):
    for index,(name,expected) in enumerate(config['asset_hashes'].items(),1):
        if progress:progress(f'正在校验模型文件 {index}/{len(config["asset_hashes"])}：{Path(name).name}')
        path=package_path(root,name)
        if not path.is_file():raise FileNotFoundError('缺少模型文件，请完整解压软件包：'+name)
        if sha(path)!=expected:raise ValueError('模型文件校验失败，请重新复制或解压：'+name)

class BirdRecognizer:
    def __init__(self,package_root,device='auto',mode='selected',progress=None):
        if mode not in ('selected','baseline'):raise ValueError('未知识别模式')
        if device not in ('auto','cuda','cpu'):raise ValueError('未知计算设备')
        self.root=Path(package_root).resolve();self.progress=progress or (lambda message:None)
        self.config=read(package_path(self.root,'configs/recognizer.json'))
        if self.config['selected_variant']!='blend50_N_ridge':raise ValueError('识别配置与冻结模型不一致')
        expected={'imgsz':640,'conf':.16,'iou':.7,'classes':[0],'verifier_threshold':.15,'detector_bypass':.35}
        if self.config['detector']!=expected:raise ValueError('第四版检测参数已被改变')
        verify_assets(self.root,self.config,self.progress)
        self.progress('正在检查计算设备…')
        import torch
        self.torch=torch;self.device_warning='';self.mode=mode
        available=torch.cuda.is_available()
        if device=='cuda' and not available:raise RuntimeError('NVIDIA GPU不可用。请安装或更新NVIDIA驱动，也可以在界面选择CPU。')
        self.device='cuda' if available and device!='cpu' else 'cpu'
        if self.device=='cuda':
            major,minor=torch.cuda.get_device_capability(0)
            arches=[int(a.removeprefix('sm_')) for a in torch.cuda.get_arch_list() if a.startswith('sm_')]
            if arches and 10*major+minor<min(arches):
                if device=='cuda':raise RuntimeError('当前显卡架构较旧，不受内置CUDA运行库支持，请改选CPU。')
                self.device='cpu';self.device_warning='当前显卡架构较旧，已选择CPU兼容模式。'
        self.numeric_mode='bf16' if self.device=='cuda' and torch.cuda.get_device_capability(0)[0]>=8 else 'fp32'
        if self.device=='cuda' and self.numeric_mode=='fp32':self.device_warning='当前显卡使用FP32兼容模式，结果可能与已验证GPU略有差异。'
        if self.device=='cpu' and not self.device_warning:self.device_warning='正在使用CPU；大模型识别会明显较慢，CPU模式未做完整准确率回归。'
        self.device_label=torch.cuda.get_device_name(0) if self.device=='cuda' else 'CPU'
        self.labels=read(package_path(self.root,self.config['labels']))
        if len(self.labels)!=1343:raise ValueError('鸟种名单不完整')
        self.detector=None;self.base=None;self.p_encoder=None;self.heads=None;self.closed=False
    def _empty_cache(self):
        if self.device=='cuda':self.torch.cuda.empty_cache()
    def _load_detector(self):
        if self.detector is not None:return
        from inference.detector import Detector
        self.progress('正在加载第四版YOLO检测模型…')
        self.detector=Detector(self.root,self.config['detector'],self.device)
    def _load_base(self):
        if self.base is not None:return
        from inference.encoders import LocalEncoder
        self._load_detector()
        self.progress('正在加载原鸟种识别模型，首次需要一些时间…')
        encoder=LocalEncoder(self.root,'bioclip2',self.device,4,self.numeric_mode)
        species=ModelE(self.root,self.config['species_e'])
        self.base=EngineN(self.device,detector=self.detector,encoder=encoder,species=species,labels=self.labels)
    def _detector_hashes(self):
        return {k:v for k,v in self.config['asset_hashes'].items() if k.startswith('models/detection/')}
    def detect_initial(self,image):
        if self.closed:raise RuntimeError('识别器已经关闭')
        if not isinstance(image,Image.Image):raise TypeError('需要PIL图像')
        image=image.convert('RGB');self._load_detector()
        if self.p_encoder is not None:self.p_encoder.model.to('cpu')
        if self.base is not None:self.base.encoder.model.to('cpu')
        self._empty_cache()
        if self.device=='cuda':self.torch.cuda.synchronize()
        begin=time.perf_counter();self.progress('正在执行第四版YOLO初检…')
        ordinary=self.detector.detect(image,mode='standard',render=False)
        if self.device=='cuda':self.torch.cuda.synchronize()
        return make_initial_packet(image,ordinary,self._detector_hashes(),self.config['detector'],
            (time.perf_counter()-begin)*1000,self.device)
    def _apply_selected(self,original,result):
        original_infos=[b['species_info'] for b in original['boxes']]
        applied=self.mode=='selected' and bool(original_infos);load_ms=0.;p_encoder_ms=0.
        if applied:
            self.base.encoder.sync();self.base.encoder.model.to('cpu');self._empty_cache()
            if self.p_encoder is None:
                from inference.encoders import LocalEncoder
                self.progress('正在加载BioCLIP 2.5高精度模型，首次需要一些时间…')
                begin=time.perf_counter();self.p_encoder=LocalEncoder(self.root,'bioclip25',self.device,2,self.numeric_mode)
                load_ms=(time.perf_counter()-begin)*1000
            self.p_encoder.model.to('cpu');self._empty_cache();self.p_encoder.model.visual.to(self.device)
            indices=[int(info['feature_index']) for info in original_infos]
            base_features=original['_base_features'][indices];crops=[original['_crops'][i] for i in indices]
            self.progress(f'正在分别识别 {len(crops)} 只鸟…')
            features,p_encoder_ms=self.p_encoder.encode_images(crops)
            if self.heads is None:self.heads=PHeads.load(self.root/'models/species/heads_p.npz')
            old=self.base.species.predict_features(base_features)
            old_names=[self.labels[int(i)]['scientific_name'] for i in old['logits'].argmax(1)]
            if old_names!=[r['name'] for r in original_infos] or old['accepted'].tolist()!=[r['accepted'] for r in original_infos]:raise ValueError('原模型复核结果不一致，停止输出')
            old_prob=softmax(old['logits'].astype(np.float64)/self.base.species.calibration['temperature'],axis=1)
            probabilities=self.heads.probabilities(features,old_prob,self.config['p_temperatures'],variants=('blend50_N_ridge',))
            infos=species_rows(probabilities['blend50_N_ridge'],self.labels,'blend50_N_ridge',original_infos)
            for box,info in zip(result['boxes'],infos):box['species_info']=info
        return applied,load_ms,p_encoder_ms
    def predict(self,image,detection_mode='normal',initial_detection=None):
        if self.closed:raise RuntimeError('识别器已经关闭')
        if not isinstance(image,Image.Image):raise TypeError('需要PIL图像')
        if detection_mode not in ('normal','tiled'):raise ValueError('未知检测方式')
        image=image.convert('RGB')
        windows=tile_windows(image.size) if detection_mode=='tiled' else []
        ordinary=None
        if initial_detection is not None:
            ordinary=validate_initial_packet(initial_detection,image,self._detector_hashes(),self.config['detector'])
        self._load_base();image=image.convert('RGB');start=time.perf_counter()
        if self.p_encoder is not None:
            self.p_encoder.model.to('cpu');self._empty_cache()
        self.base.encoder.model.to(self.device)
        self.progress('正在整图检测并分块补检小鸟…' if windows else '正在整图检测鸟的位置并补检小目标…')
        original=self.base.predict(image,'rescue1280',initial_detection=ordinary,tile_windows=windows)
        tile_original=original.pop('_tile_result',None);result=deepcopy(original)
        original_infos=[b['species_info'] for b in original['boxes']]
        p_begin=time.perf_counter();applied,load_ms,p_encoder_ms=self._apply_selected(original,result)
        p_classification_ms=(time.perf_counter()-p_begin)*1000
        for before,after in zip(original['boxes'],result['boxes']):
            assert {k:v for k,v in before.items() if k!='species_info'}=={k:v for k,v in after.items() if k!='species_info'}
            assert all(before['species_info'][k]==after['species_info'][k] for k in ('accepted','birdness_score'))
        normal_boxes=deepcopy(result['boxes']);tile_p_ms=0.;tile_e_ms=0.;tile_detector_ms=0.
        if tile_original is not None:
            tile_result=deepcopy(tile_original);p_begin=time.perf_counter()
            tile_applied,tile_load_ms,tile_p_ms=self._apply_selected(tile_original,tile_result)
            p_classification_ms+=(time.perf_counter()-p_begin)*1000
            applied=applied or tile_applied;load_ms+=tile_load_ms;p_encoder_ms+=tile_p_ms
            offset=len(result['_crops']);seen=set()
            for row in tile_result['candidate_records']:
                if row['feature_index'] is not None:row['feature_index']+=offset
                info=row['box'].get('species_info')
                if info is not None and id(info) not in seen:info['feature_index']+=offset;seen.add(id(info))
            for box in tile_result['boxes']:
                info=box['species_info']
                if id(info) not in seen:info['feature_index']+=offset;seen.add(id(info))
            result['boxes'].extend(tile_result['boxes']);result['added'].extend(tile_result['boxes'])
            result['candidate_records'].extend(tile_result['candidate_records'])
            result['_crops'].extend(tile_result['_crops'])
            for key in ('_base_features','_adapted_features'):result[key]=np.concatenate([result[key],tile_result[key]],axis=0)
            result['tile_scan']={k:v for k,v in tile_result['scan'].items() if k!='verified'}
            tile_e_ms=tile_result['classification_ms'];tile_detector_ms=tile_result['scan']['detector_ms']
            tile_baseline=deepcopy([b['species_info'] for b in tile_original['boxes']])
            for info in tile_baseline:info['feature_index']+=offset
            original_infos=original_infos+tile_baseline
        if result['boxes'][:len(normal_boxes)]!=normal_boxes:raise AssertionError('分块补检改变了普通模式结果')
        if result['primary_index'] is None and result['boxes']:
            result['primary_index']=max(range(len(result['boxes'])),key=lambda i:result['boxes'][i]['confidence'])
        result.update(detection_mode=detection_mode,tiled=bool(windows),tile_windows=len(windows),
            tile_candidates=tile_original['scan']['raw_candidates'] if tile_original else 0,
            tile_added=len(result['boxes'])-len(normal_boxes),normal_prefix_length=len(normal_boxes),
            normal_output_preserved=True,initial_detection_reused=initial_detection is not None,
            packet_initial_detection_ms=initial_detection['elapsed_ms'] if initial_detection else None,
            detection_origins=['ordinary' if i<original['ordinary_prefix_length'] else 'rescue1280' for i in range(len(normal_boxes))]+['tile']*(len(result['boxes'])-len(normal_boxes)))
        result['detector_passes']+=len(windows)-(1 if initial_detection is not None else 0)
        result['classification_ms']+=tile_e_ms
        common={x['scientific_name']:x.get('common_name') or x['scientific_name'] for x in self.labels}
        for b in result['boxes']:
            for top in b['species_info']['top5']:top['common_name']=common.get(top['name'],top['name'])
        self.base.encoder.sync()
        result.update(candidate_id='blend50_N_ridge' if self.mode=='selected' else 'baseline',runtime_mode=self.mode,
            device_label=self.device_label,numeric_mode=self.numeric_mode,device_warning=self.device_warning,
            p_classifier_applied=applied,p_encoder_lazy_load_ms=load_ms,p_encoder_ms=p_encoder_ms,
            total_ms=(time.perf_counter()-start)*1000,baseline_labels=deepcopy(original_infos),
            original_N_acceptance_unchanged=True,selected_model_sha256=self.config['frozen_P_selection_sha256'],
            accuracy_reference_numeric_mode='cuda_bf16',physical_pi_test=False)
        result['stage_timings_ms']={'initial_detection':result['ordinary_detector_ms'],
            'full_image_supplement':result['extra_detector_ms'],'tiled_supplement':tile_detector_ms,
            'classification':result['classification_ms']+p_classification_ms,'total':result['total_ms']}
        result['species_group_timings_ms']={'normal_E':original['classification_ms'],'tile_E':tile_e_ms,
            'normal_P_encode':p_encoder_ms-tile_p_ms,'tile_P_encode':tile_p_ms}
        self.progress('识别完成，正在保存标框图和逐鸟裁切…')
        return result
    def close(self):
        if self.closed:return
        if self.p_encoder is not None:self.p_encoder.close()
        if self.base is not None:self.base.close()
        self.p_encoder=None;self.base=None;self.detector=None;self.heads=None;self.closed=True;gc.collect();self._empty_cache()

def annotate(image,boxes):
    output=image.copy();draw=ImageDraw.Draw(output);size=max(14,min(30,min(image.size)//30));font=ImageFont.load_default()
    path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/msyh.ttc'
    if path.exists():font=ImageFont.truetype(str(path),size)
    for b in boxes:
        info=b['species_info'];color='#1A9D72' if info['accepted'] else '#D99100'
        label=b['bird_id']+' '+info['display_name'];x1,y1,x2,y2=b['xyxy']
        draw.rectangle((x1,y1,x2,y2),outline=color,width=max(2,min(image.size)//250))
        bb=draw.textbbox((0,0),label,font=font);w,h=bb[2]-bb[0],bb[3]-bb[1];tx=max(0,min(x1,image.width-w-8));ty=max(0,y1-h-8)
        draw.rectangle((tx,ty,tx+w+8,ty+h+8),fill=color);draw.text((tx+4,ty+4-bb[1]),label,font=font,fill='white')
    return output

def save_result(image,result,source,output_root):
    folder=Path(output_root).resolve()/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True,exist_ok=False)
    image=image.convert('RGB');public=deepcopy({k:v for k,v in result.items() if not k.startswith('_')})
    image.save(folder/'snapshot.png')
    for index,box in enumerate(public['boxes'],1):
        box['bird_id']=f'bird_{index:03d}';box['status']='classified'
        crop=result['_crops'][box['species_info']['feature_index']]
        expected=square_context(image,box['xyxy'])
        if crop.size!=expected.size or crop.tobytes()!=expected.tobytes():raise ValueError('鸟图裁切与检测框不一致')
        x1,y1,x2,y2=box['xyxy'];bounds=(max(0,math.floor(x1)),max(0,math.floor(y1)),min(image.width,math.ceil(x2)),min(image.height,math.ceil(y2)))
        crop_path=folder/(box['bird_id']+'.png');tight_path=folder/(box['bird_id']+'_tight.png')
        crop.save(crop_path);image.crop(bounds).save(tight_path)
        box.update(crop_path=str(crop_path),crop_sha256=sha(crop_path),tight_crop_path=str(tight_path),tight_crop_sha256=sha(tight_path),N_and_P_use_same_crop=True)
    annotated=folder/'annotated.png';annotate(image,public['boxes']).save(annotated)
    public.update(event_dir=str(folder),snapshot=str(folder/'snapshot.png'),annotated_path=str(annotated),source=str(source),
        created_utc=datetime.now(timezone.utc).isoformat(),all_final_boxes_cropped=True,classifier_input='each exact 1.2 square bird crop')
    if Path(source).is_file():public['source_sha256']=sha(source)
    public['result_json']=str(folder/'result.json')
    (folder/'result.json').write_text(json.dumps(public,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return public
