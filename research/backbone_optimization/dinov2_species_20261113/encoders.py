"""Frozen image encoders with separate transforms for a shared RGB crop."""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PACKAGE = Path(r'D:\项目\鸟类识别\识别模型\v1\v1.1\NVIDIA')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class DinoEncoder:
    dimension = 384

    def __init__(self, device):
        assets = HERE / 'assets'
        checkpoint = assets / 'dinov2_vits14_pretrain.pth'
        if sha(checkpoint) != read(assets / 'dinov2.json')['weight_sha256']:
            raise ValueError('DINOv2 checkpoint hash differs')
        os.environ['XFORMERS_DISABLED'] = '1'
        sys.path.insert(0, str(assets / 'dinov2_source'))
        from dinov2.hub.backbones import dinov2_vits14
        from torchvision import transforms as T
        self.preprocess = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224), T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])
        self.device = device
        self.model = dinov2_vits14(pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.model.requires_grad_(False).eval().to(device)

    @torch.inference_mode()
    def encode(self, images, patches=False):
        batch = torch.stack([self.preprocess(image.convert('RGB')) for image in images]).to(self.device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device == 'cuda'):
            result = self.model.forward_features(batch)
        cls = F.normalize(result['x_norm_clstoken'].float(), dim=-1).cpu().numpy()
        if patches:
            return cls, result['x_norm_patchtokens'].float().cpu().numpy()
        return cls


def bioclip(device, kind='bioclip2', batch=4):
    from inference.encoders import LocalEncoder
    cfg = read(PACKAGE / 'configs/recognizer.json')
    name = f'models/{kind}/open_clip_model.safetensors'
    if sha(PACKAGE / name) != cfg['asset_hashes'][name]:
        raise ValueError('BioCLIP checkpoint hash differs')
    return LocalEncoder(PACKAGE, kind, device, batch, 'bf16')
