"""A genuinely trainable DINO final block, with the original species head."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from encoders import DinoEncoder
from fusion import SpeciesNet
from common import CLASSES, warmstart


class TailSpeciesNet(nn.Module):
    def __init__(self,encoder,head_state,train_tail):
        super().__init__()
        assert len(encoder.model.blocks)==12 and not encoder.model.chunked_blocks
        assert encoder.model.num_register_tokens==0
        self.block=deepcopy(encoder.model.blocks[-1])
        self.norm=deepcopy(encoder.model.norm)
        self.head=SpeciesNet('fusion',len(CLASSES),head_state['projection'],head_state['dropout'])
        self.head.load_state_dict(head_state['model'],strict=True)
        self.head.requires_grad_(True)
        self.block.requires_grad_(train_tail)
        self.norm.requires_grad_(train_tail)
        self.train_tail=train_tail

    def visual(self,tokens):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=tokens.is_cuda):
            x=self.norm(self.block(tokens))[:,0]
        return F.normalize(x.float(),dim=-1)

    def forward(self,bio,tokens):
        # Head math remains FP32, matching the prior cached-feature experiment.
        return self.head(bio.float(),self.visual(tokens))


def build(seed,train_tail,device='cuda'):
    encoder=DinoEncoder('cpu')
    _,checkpoint=warmstart(seed)
    result=TailSpeciesNet(encoder,checkpoint,train_tail).to(device)
    del encoder
    return result


def restore(path,device='cuda'):
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['global_ids']==CLASSES.tolist()
    model=build(state['seed'],state['train_tail'],device)
    model.load_state_dict(state['model'],strict=True)
    return model.eval(),state
