import importlib.metadata
import importlib.util
import math
import sys
import numpy as np
from audit_project import HERE,PACKAGE,DATA,read,save,sha

names=['torch','torchvision','open_clip_torch','timm','safetensors','huggingface_hub',
       'ultralytics','numpy','scipy','scikit-learn','Pillow','matplotlib','transformers','xformers']
versions={}
for name in names:
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions[name]=None
import torch
import open_clip
from open_clip.transform import PreprocessCfg,image_transform_v2
from ultralytics import YOLO

details={}
for kind,name in [('bioclip2','ViT-L-14'),('bioclip25','ViT-H-14')]:
    config=read(PACKAGE/f'models/{kind}/open_clip_config.json')
    assert config['model_cfg']==open_clip.get_model_config(name)
    with torch.device('meta'):
        model=open_clip.create_model(name,pretrained=None,pretrained_image=False,pretrained_text=False,device='meta')
    details[kind]={'visual_parameters':sum(p.numel() for p in model.visual.parameters()),
                   'total_model_parameters':sum(p.numel() for p in model.parameters()),
                   'count_basis':'Empty meta structure only; no real encoder created or checkpoint loaded',
                   'preprocess':str(image_transform_v2(PreprocessCfg(size=224,**config['preprocess_cfg']),is_train=False))}
    del model
detector=YOLO(str(PACKAGE/'models/detection/detector_reference.pt'))
details['detector']={'task':detector.task,'yaml':detector.model.yaml,'names':detector.names,
                     'checkpoint_path':str(PACKAGE/'models/detection/detector_reference.pt'),
                     'original_training_args_model':getattr(detector.model,'args',{}).get('model'),
                     'parameters':sum(p.numel() for p in detector.model.parameters()),
                     'cpu_only_checkpoint_inspection':True,'no_inference':True}
del detector
heads={}
for relative in ['models/species/adapted_head.npz','models/species/heads_p.npz','models/species/original_text.npz']:
    with np.load(PACKAGE/relative,allow_pickle=False) as z:
        heads[relative]={k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files}
        for k in ['supported_ids','ids']:
            if k in z.files:heads[relative]['supported_species_ids']=z[k].tolist()
save('runtime_probe.json',{'python':sys.version,'versions':versions,'torch_cuda_build':torch.version.cuda,
     'cuda_context_initialized':torch.cuda.is_initialized(),'model_details':details,'head_arrays':heads,
     'DINO_weights_loaded':False,'BioCLIP_weights_loaded_into_model':False,'training_started':False})
print('RUNTIME_PROBE_COMPLETE',versions,flush=True)
