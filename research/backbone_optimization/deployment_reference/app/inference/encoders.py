import json,time,gc
import numpy as np

def load_visual_model(name,checkpoint,preprocess_config):
    import torch,open_clip
    from safetensors import safe_open
    from torch.nn import functional as F
    pp=preprocess_config
    with torch.device('meta'):
        full,_,preprocess=open_clip.create_model_and_transforms(name,
            pretrained=None,pretrained_image=False,pretrained_text=False,
            device='meta',precision='fp32',image_mean=pp['mean'],image_std=pp['std'],
            image_interpolation=pp['interpolation'],image_resize_mode=pp['resize_mode'])
    visual=full.visual;del full
    expected=visual.state_dict()
    for key,buffer in visual.named_buffers():
        if buffer.is_meta and key not in expected:
            raise ValueError('视觉模型含未提供数值的非持久缓冲区：'+key)
    with safe_open(str(checkpoint),framework='pt',device='cpu',backend='pread') as file:
        keys={key[len('visual.'):] for key in file.keys() if key.startswith('visual.')}
        if keys!=set(expected):
            raise ValueError('视觉权重键与冻结结构不一致：missing='+str(sorted(set(expected)-keys))+'; extra='+str(sorted(keys-set(expected))))
        for key,reference in expected.items():
            if tuple(file.get_slice('visual.'+key).get_shape())!=tuple(reference.shape):
                raise ValueError('视觉权重形状与冻结结构不一致：'+key)
        state={key:file.get_tensor('visual.'+key) for key in expected}
        for key,value in state.items():
            if value.dtype!=expected[key].dtype or (value.is_floating_point() and value.dtype!=torch.float32):
                raise ValueError('视觉权重精度与冻结FP32模型不一致：'+key)
        visual.load_state_dict(state,strict=True,assign=True)
    del state,expected
    for key,value in list(visual.named_parameters())+list(visual.named_buffers()):
        if value.is_meta:raise ValueError('视觉模型仍含未加载参数：'+key)
    class VisualOnly(torch.nn.Module):
        def __init__(self,tower):
            super().__init__();self.visual=tower
        def encode_image(self,image,normalize=False):
            features=self.visual(image)
            return F.normalize(features,dim=-1) if normalize else features
    model=VisualOnly(visual).eval().requires_grad_(False)
    return model,preprocess

class LocalEncoder:
    def __init__(self,root,kind,device='cuda',batch_size=4,numeric_mode='bf16'):
        import torch,open_clip
        self.torch=torch;self.device=device;self.batch_size=batch_size;self.numeric_mode=numeric_mode
        folder=root/'models'/kind
        config=json.loads((folder/'open_clip_config.json').read_text(encoding='utf-8'))
        name='ViT-L-14' if kind=='bioclip2' else 'ViT-H-14';self.dimension=768 if kind=='bioclip2' else 1024
        if config['model_cfg']!=open_clip.get_model_config(name):raise ValueError('运行库中的模型结构与本地模型配置不同')
        torch.set_num_threads(4)
        if device=='cuda':torch.backends.cuda.matmul.allow_tf32=False
        self.model,self.preprocess=load_visual_model(name,folder/'open_clip_model.safetensors',config['preprocess_cfg'])
        self.model.visual.to(device)
    def sync(self):
        if self.device=='cuda':self.torch.cuda.synchronize()
    def encode_images(self,images):
        torch=self.torch;features=[];self.sync();start=time.perf_counter()
        with torch.inference_mode():
            for begin in range(0,len(images),self.batch_size):
                tensor=torch.stack([self.preprocess(image.convert('RGB')) for image in images[begin:begin+self.batch_size]]).to(self.device)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device=='cuda' and self.numeric_mode=='bf16'):
                    value=self.model.encode_image(tensor,normalize=True)
                features.append(value.float().cpu().numpy())
        self.sync();elapsed=(time.perf_counter()-start)*1000
        return np.concatenate(features) if features else np.empty((0,self.dimension),np.float32),elapsed
    def close(self):
        self.model=None;gc.collect()
        if self.device=='cuda':self.torch.cuda.empty_cache()
