"""Exercise both official weights and verify an actual head update and reload."""
import hashlib
import io
import time
import numpy as np
import torch
from data import HERE, PACKAGE, CLASSES, crop, support_rows, save
from encoders import DinoEncoder, bioclip
from fusion import SpeciesNet


def fingerprint(model):
    digest = hashlib.sha256()
    for value in model.state_dict().values():
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


torch.set_num_threads(4)
torch.manual_seed(131)
assert torch.cuda.is_available()
rows = support_rows()[:2]
images = [crop(row) for row in rows]
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
bio = bioclip('cuda')
dino = DinoEncoder('cuda')
before = [fingerprint(model) for model in (bio.model,dino.model)]
x = torch.from_numpy(bio.encode_images(images)[0]).cuda()
global_values, local = dino.encode(images, patches=True)
y = torch.from_numpy(global_values).cuda()
assert x.shape == (2,768) and y.shape == (2,384) and local.shape == (2,256,384)
assert not any(p.requires_grad for model in (bio.model,dino.model) for p in model.parameters())
assert not bio.model.training and not dino.model.training
assert len(CLASSES) == 80 and len(set(CLASSES)) == 80
labels = torch.tensor(np.searchsorted(CLASSES, [r['target'] for r in rows]), device='cuda')
assert np.array_equal(CLASSES[labels.cpu().numpy()], [r['target'] for r in rows])
head = SpeciesNet('fusion', len(CLASSES)).cuda()
optimizer = torch.optim.AdamW(head.parameters(), lr=.0003)
old_head = {key:value.clone() for key,value in head.state_dict().items()}
out = head(x,y)
assert out.shape == (2,80)
loss = torch.nn.functional.cross_entropy(out, labels)
assert torch.isfinite(loss)
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in head.parameters())
optimizer.step()
assert any(not torch.equal(old_head[key], value) for key,value in head.state_dict().items())
assert all(p.grad is None for model in (bio.model,dino.model) for p in model.parameters())
assert before == [fingerprint(model) for model in (bio.model,dino.model)]
head.eval()
stream = io.BytesIO()
torch.save({'state':head.state_dict(),'classes':CLASSES.tolist(),'optimizer':optimizer.state_dict(),'epoch':1}, stream)
stream.seek(0)
state = torch.load(stream, weights_only=True)
reloaded = SpeciesNet('fusion', len(CLASSES)).cuda().eval()
reloaded.load_state_dict(state['state'], strict=True)
with torch.no_grad():
    assert torch.equal(head(x,y),reloaded(x,y))
assert state['classes'] == CLASSES.tolist()
save(HERE/'smoke_result.json', {'passed':True,'real_crops':2,'bio_shape':list(x.shape),'dino_shape':list(y.shape),
    'patch_shape':list(local.shape),'finite_loss':float(loss.detach()),'frozen_parameter_hashes_equal':True,
    'head_has_gradients_and_updates':True,'checkpoint_roundtrip_equal':True,'class_indices_verified':True,
    'cuda_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'seconds':time.perf_counter()-started})
print('REAL GPU SMOKE PASSED', flush=True)
