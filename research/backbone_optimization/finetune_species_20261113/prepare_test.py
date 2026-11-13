"""Extract untouched test inputs only after validation model selection is frozen."""
import argparse
import gc
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import torch
from torch.nn import functional as F
from common import HERE,OLD,PACKAGE,IDS,read,save,sha,crop,load_npz
from encoders import DinoEncoder,bioclip


def npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.npz')
    np.savez_compressed(temp,**arrays)
    temp.replace(path)


def permission():
    selection=HERE/'selection.json'
    assert selection.exists(),'Freeze all validation selections before test inference'
    manifest=HERE/'holdout_data/final_manifest.json'
    inputs=HERE/'holdout_data/final_test.json'
    qa=HERE/'holdout_data/input_qa.json'
    checked=read(qa)
    assert checked['status'] in ('passed','passed_with_notes')
    assert checked['reviewed_before_inference'] is True
    assert checked['inference_can_start'] is True
    assert checked['any_model_inference_performed'] is False
    assert checked['manifest_sha256']==sha(manifest)
    assert checked['test_sha256']==sha(inputs)
    rows=read(inputs)
    assert len(rows)==checked['images']
    return rows,{'selection_sha256':sha(selection),'holdout_manifest_sha256':sha(manifest),
                 'holdout_rows_sha256':sha(inputs),'input_qa_sha256':sha(qa),
                 'prepare_test_sha256':sha(HERE/'prepare_test.py')}


def detect(device):
    from engine import BirdRecognizer
    rows,bindings=permission()
    folder=HERE/'features'
    folder.mkdir(exist_ok=True)
    destination=folder/'test.json'
    if destination.exists():
        assert read(folder/'test_input_binding.json')==bindings
        assert load_npz(folder/'test_bio.npz')['rows_sha'].item()==sha(destination)
        return
    checkpoints=folder/'partial'
    checkpoints.mkdir(exist_ok=True)
    worker=BirdRecognizer(PACKAGE,device=device,mode='baseline')
    materialized=[]
    features=[]
    try:
        for index,row in enumerate(rows):
            key=hashlib.sha256(row['key'].encode()).hexdigest()[:24]
            checkpoint=checkpoints/(key+'.json')
            vector=checkpoints/(key+'.npy')
            if checkpoint.exists() and vector.exists():
                saved=read(checkpoint)
                assert saved['source_sha256']==row['sha256']
                assert saved['binding']==bindings
                materialized.append(saved['row'])
                features.append(np.load(vector,allow_pickle=False))
                continue
            assert sha(row['path'])==row['sha256']
            with Image.open(row['path']) as image:
                rgb=ImageOps.exif_transpose(image).convert('RGB')
            result=worker.predict(rgb,detection_mode='normal')
            selected=result['primary_index']
            box=None if selected is None else result['boxes'][selected]
            new_row={**row,'target':IDS[row['scientific_name']],
                     'bbox':None if box is None else box['xyxy'],
                     'accepted':False if box is None else bool(box['species_info']['accepted']),
                     'detector_boxes':[candidate['xyxy'] for candidate in result['boxes']],
                     'detector_mode':'normal','primary_index':selected}
            feature=np.zeros(768,np.float32) if box is None else np.asarray(
                result['_base_features'][box['species_info']['feature_index']],dtype=np.float32)
            assert feature.shape==(768,) and np.isfinite(feature).all()
            np.save(vector,feature,allow_pickle=False)
            save(checkpoint,{'row':new_row,'source_sha256':row['sha256'],'binding':bindings})
            materialized.append(new_row)
            features.append(feature)
            if (index+1)%20==0 or index+1==len(rows):
                print('Frozen YOLO normal',index+1,'/',len(rows),flush=True)
    finally:
        worker.close()
        del worker
        gc.collect()
        if device=='cuda':
            torch.cuda.empty_cache()
    save(destination,materialized)
    npz(folder/'test_bio.npz',features=np.stack(features),rows_sha=np.array(sha(destination)))
    save(folder/'test_input_binding.json',bindings)


def encode_bio25(device):
    rows=read(HERE/'features/test.json')
    dest=HERE/'features/test_bio25.npz'
    if dest.exists():
        assert load_npz(dest)['rows_sha'].item()==sha(HERE/'features/test.json')
        return
    model=bioclip(device,'bioclip25',4)
    present=[i for i,row in enumerate(rows) if row['bbox'] is not None]
    output=np.zeros((len(rows),1024),np.float32)
    try:
        for begin in range(0,len(present),4):
            index=present[begin:begin+4]
            output[index]=model.encode_images([crop(rows[i]) for i in index])[0]
    finally:
        model.close()
        del model
        gc.collect()
    npz(dest,features=output,rows_sha=np.array(sha(HERE/'features/test.json')))
    print('BioCLIP2.5 test features',len(present),flush=True)


def encode_dino(device):
    rows_path=HERE/'features/test.json'
    rows=read(rows_path)
    dest=HERE/'tokens/test.npy'
    metadata=HERE/'tokens/test.json'
    if metadata.exists():
        assert read(metadata)['rows_sha256']==sha(rows_path)
        assert read(metadata)['tokens_sha256']==sha(dest)
        assert load_npz(HERE/'features/test_dino.npz')['rows_sha'].item()==sha(rows_path)
        return
    encoder=DinoEncoder(device)
    present=[i for i,row in enumerate(rows) if row['bbox'] is not None]
    pending=dest.with_suffix('.pending.npy')
    tokens=np.lib.format.open_memmap(pending,mode='w+',dtype=np.float32,shape=(len(rows),257,384))
    tokens[:]=0
    final=np.zeros((len(rows),384),np.float32)
    for begin in range(0,len(present),16):
        index=present[begin:begin+16]
        batch=torch.stack([encoder.preprocess(crop(rows[i])) for i in index]).to(device)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
            x=encoder.model.prepare_tokens_with_masks(batch)
            for block in encoder.model.blocks[:-1]:
                x=block(x)
        tokens[index]=x.float().cpu().numpy()
    # Bio-tail training consumes the earlier frozen DINO features (batch=8).
    # Keep that encoder path distinct from the batch=16 trainable-prefix cache.
    for begin in range(0,len(present),8):
        index=present[begin:begin+8]
        final[index]=encoder.encode([crop(rows[i]) for i in index])
    tokens.flush()
    del tokens,encoder
    gc.collect()
    if device=='cuda':
        torch.cuda.empty_cache()
    pending.replace(dest)
    npz(HERE/'features/test_dino.npz',features=final,rows_sha=np.array(sha(rows_path)))
    save(metadata,{'rows_sha256':sha(rows_path),'tokens_sha256':sha(dest),
         'prefix_binding_sha256':read(HERE/'tokens/manifest.json')['binding_sha256'],
         'images':len(rows),'nonempty_crops':len(present),'dtype':'float32','tokens':[257,384],
         'split':'test','heldout_input_binding_sha256':sha(HERE/'features/test_input_binding.json')})
    print('DINO test prefix features',len(present),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--bio-prefix',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    permission()
    detect(args.device)
    encode_bio25(args.device)
    encode_dino(args.device)
    if args.bio_prefix:
        from bio_tail import prepare_test
        prepare_test(args.device)
    _,bindings=permission()
    outputs={str(path.relative_to(HERE)):sha(path) for path in [
        HERE/'features/test.json',HERE/'features/test_bio.npz',HERE/'features/test_bio25.npz',
        HERE/'features/test_dino.npz',HERE/'tokens/test.npy',HERE/'tokens/test.json']}
    save(HERE/'features/test_contract.json',{'inputs':bindings,'outputs_sha256':outputs,
         'original_detection_and_species_gate_unchanged':True,'model_selection_used_test':False})
    print('Frozen test input features complete',flush=True)


if __name__=='__main__':
    main()
