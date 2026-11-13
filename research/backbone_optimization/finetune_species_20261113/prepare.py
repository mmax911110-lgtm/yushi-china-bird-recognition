"""Cache only the frozen prefix, preserving the exact old fixed-crop input."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,OLD,CLASSES,read,save,sha,state_sha,crop,load_npz,warmstart
from encoders import DinoEncoder


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda',choices=('cuda','cpu'))
    args=parser.parse_args()
    torch.set_num_threads(4)
    config=read(HERE/'config.json')
    splits=config['train_splits']+config['validation']
    folder=HERE/'tokens'
    folder.mkdir(parents=True,exist_ok=True)
    binding={relative:sha(OLD/relative) for split in splits for relative in
             (f'features/{split}.json',f'features/{split}_bio.npz',f'features/{split}_dino.npz')}
    binding.update({str(path):sha(path) for seed in config['seeds'] for path,_ in [warmstart(seed)]})
    binding['config_sha256']=sha(HERE/'config.json')
    binding['dino_weights']=sha(OLD/'assets/dinov2_vits14_pretrain.pth')
    binding_path=folder/'binding.json'
    if binding_path.exists():
        assert read(binding_path)==binding
    else:
        save(binding_path,binding)
    encoder=DinoEncoder(args.device)
    assert len(encoder.model.blocks)==12 and not encoder.model.chunked_blocks
    prefix_before=state_sha(encoder.model)
    all_differences={}
    for split in splits:
        dest=folder/(split+'.npy')
        metadata=folder/(split+'.json')
        rows_path=OLD/'features'/(split+'.json')
        rows=read(rows_path)
        if metadata.exists():
            contract=read(metadata)
            assert sha(dest)==contract['tokens_sha256']
            assert contract['rows_sha256']==sha(rows_path)
            assert contract['prefix_binding_sha256']==sha(binding_path)
            all_differences[split]=contract['reconstruction_max_abs_error']
            continue
        source=load_npz(OLD/'features'/(split+'_dino.npz'))
        assert source['rows_sha'].item()==sha(rows_path)
        pending=dest.with_suffix('.pending.npy')
        output=np.lib.format.open_memmap(pending,mode='w+',dtype=np.float32,shape=(len(rows),257,384))
        output[:]=0
        present=[i for i,row in enumerate(rows) if row['bbox'] is not None]
        difference=0.
        for begin in range(0,len(present),16):
            index=present[begin:begin+16]
            images=[crop(rows[i]) for i in index]
            batch=torch.stack([encoder.preprocess(image) for image in images]).to(args.device)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.device=='cuda'):
                tokens=encoder.model.prepare_tokens_with_masks(batch)
                for block in encoder.model.blocks[:-1]:
                    tokens=block(tokens)
                assert tokens.shape==(len(index),257,384)
                assert torch.isfinite(tokens).all()
                reference=F.normalize(encoder.model.norm(encoder.model.blocks[-1](tokens))[:,0].float(),dim=-1)
            values=tokens.float().cpu().numpy()
            output[index]=values
            difference=max(difference,float(np.max(np.abs(reference.cpu().numpy()-source['features'][index]))))
            if begin%128==0:
                print(split,begin,'/',len(present),'max reconstruction delta',difference,flush=True)
        output.flush()
        del output
        pending.replace(dest)
        # Batch-dependent BF16 GEMM may differ slightly from the previous batch=8 cache.
        assert difference<.015,(split,difference)
        save(metadata,{'rows_sha256':sha(rows_path),'tokens_sha256':sha(dest),
             'prefix_binding_sha256':sha(binding_path),'images':len(rows),'nonempty_crops':len(present),
             'reconstruction_max_abs_error':difference,'dtype':'float32','tokens':[257,384]})
        all_differences[split]=difference
    assert state_sha(encoder.model)==prefix_before
    save(folder/'manifest.json',{'binding_sha256':sha(binding_path),'splits':splits,
         'source_checkpoint_sha256':binding['dino_weights'],'unchanged_full_backbone_sha256':prefix_before,
         'prefix_cache_reconstructs_old_DINO_features_max_error':all_differences,
         'storage':'Float32 raw CLS+256patch tokens, after frozen blocks0..10, before trainable block11/norm',
         'no_augmentation':True,'test_images_encoded':False})
    print('Frozen prefix caches complete',all_differences,flush=True)


if __name__=='__main__':
    main()
