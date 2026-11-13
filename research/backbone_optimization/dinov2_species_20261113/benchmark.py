"""Measure warm single-crop latency and CUDA peaks on validation images."""
import time
import gc
import numpy as np
import torch
from data import HERE, crop, read, save
from encoders import bioclip, DinoEncoder
from inference.models_e import ModelE
from train import load_head

device = 'cuda'
torch.set_num_threads(4)
selection = read(HERE/'runs/selection.json')
selected = next(c for c in selection['candidates'] if c['mode']=='fusion' and c['group']=='expanded' and c['form']=='closed')
rows = [r for r in read(HERE/'features/n_val.json') if r['bbox'] is not None]
images = [crop(rows[i]) for i in np.linspace(0,len(rows)-1,16,dtype=int)]
result = {}
for variant in ('E0_BioCLIP2','E1_DINOv2','E2_global_fusion'):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    e0 = variant=='E0_BioCLIP2'
    e1 = variant=='E1_DINOv2'
    b = None if e1 else bioclip(device)
    d = None if e0 else DinoEncoder(device)
    if e0:
        from encoders import PACKAGE
        head=ModelE(PACKAGE,read(PACKAGE/'configs/recognizer.json')['species_e'])
    else:
        mode='dino' if e1 else 'fusion'
        spec=next(c for c in selection['candidates'] if c['mode']==mode and c['group']=='expanded' and c['form']=='closed')
        heads=[load_head(r['checkpoint'],device) for r in spec['models']]

    def once(image):
        x = None if b is None else b.encode_images([image])[0]
        if e0:
            return head.predict_features(x)
        y=d.encode([image])
        xb=None if x is None else torch.from_numpy(x).cuda()
        yb=torch.from_numpy(y).cuda()
        with torch.inference_mode():
            return torch.stack([model(xb,yb) for model in heads]).mean(0).softmax(1).cpu()

    for image in images[:4]:
        once(image)
    torch.cuda.reset_peak_memory_stats()
    elapsed=[]
    for _ in range(3):
        for image in images:
            torch.cuda.synchronize()
            started=time.perf_counter()
            once(image)
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter()-started)*1000.)
    result[variant]={'samples':len(elapsed),'median_ms':float(np.median(elapsed)),'p95_ms':float(np.percentile(elapsed,95)),
        'allocated_peak_MiB':torch.cuda.max_memory_allocated()/2**20,'reserved_peak_MiB':torch.cuda.max_memory_reserved()/2**20,
        'visual_parameters':sum(p.numel() for encoder in (b,d) if encoder for p in encoder.model.parameters()),
        'trainable_classifier_parameters':0 if e0 else sum(p.numel() for model in heads for p in model.parameters())}
    if b is not None:
        b.close()
    del b,d
    if not e0:
        del heads
    print(variant,result[variant],flush=True)
save(HERE/'runs/benchmark.json', {'GPU':torch.cuda.get_device_name(0),'batch':1,'crop_preprocessing_included':True,
    'load_decode_and_yolo_excluded':True,'E0_classifier_executed_on_cpu':True,'E1_E2_three_seed_ensemble':True,'results':result})
