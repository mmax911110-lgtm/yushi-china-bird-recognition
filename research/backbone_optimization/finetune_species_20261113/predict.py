"""Experimental whole-image YOLO -> per-candidate crop -> frozen species model.

Only candidates already frozen in this experiment's selection.json are accepted.
Normal fourth-version detections, primary index and original E gates are preserved.
Large visual encoders are loaded sequentially; this does not implement tiled rescue.
"""
import argparse
from copy import deepcopy
import gc
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.special import softmax
import torch

from common import HERE, PACKAGE, CLASSES, LABELS, read, save, sha
from inference.engine_n import square_context


def release(device):
    gc.collect()
    if device=='cuda':
        torch.cuda.empty_cache()


def select_candidate(identifier):
    path=HERE/'selection.json'
    if not path.exists():
        raise ValueError('selection.json is not frozen yet; finish validation selection first.')
    selection=read(path)
    from scoring import assert_selection_inputs
    assert_selection_inputs(selection)
    choices=[spec for spec in selection['candidates'] if spec['id']==identifier]
    if len(choices)!=1:
        available=', '.join(spec['id'] for spec in selection['candidates'])
        raise ValueError('Candidate must match one frozen spec ID. Available: '+available)
    spec=choices[0]
    if spec['form'] not in ('closed','full_residual') or spec['architecture'] not in ('dinov2_tail','bioclip_tail'):
        raise ValueError('Unsupported frozen candidate architecture/form')
    if len(spec['models'])!=3:
        raise ValueError('Expected the frozen three-seed ensemble')
    if len({item['seed'] for item in spec['models']})!=3:
        raise ValueError('Duplicate ensemble seeds')
    if not np.isfinite(spec['temperature']) or spec['temperature']<=0:
        raise ValueError('Invalid frozen validation temperature')
    for item in spec['models']:
        if item.get('architecture')!=spec['architecture']:
            raise ValueError('Mixed backbone architectures in a single candidate')
        if sha(item['checkpoint'])!=item['checkpoint_sha256']:
            raise ValueError('Candidate checkpoint changed: '+item['checkpoint'])
    return selection,spec,sha(path)


def bio_prefix(encoder,crops,device):
    """Match bio_tail._prepare_one: OpenCLIP embeds + blocks0..22 in BF16."""
    from bio_tail import assert_visual, source_binding
    source_binding()
    visual=encoder.model.visual
    assert_visual(visual)
    pieces=[]
    for begin in range(0,len(crops),4):
        batch=torch.stack([encoder.preprocess(image.convert('RGB')) for image in crops[begin:begin+4]]).to(device)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
            tokens=visual._embeds(batch)
            for block in visual.transformer.resblocks[:-1]:
                tokens=block(tokens,attn_mask=None)
            assert tuple(tokens.shape[1:])==(257,1024) and torch.isfinite(tokens).all()
            assert tokens.dtype==(torch.bfloat16 if device=='cuda' else torch.float32)
        pieces.append(tokens.float().cpu().numpy())
    return np.concatenate(pieces).astype(np.float32,copy=False)


def normal_detection(image,device,need_bio_prefix):
    from engine import BirdRecognizer
    worker=BirdRecognizer(PACKAGE,device=device,mode='baseline')
    try:
        result=worker.predict(image,detection_mode='normal')
        boxes=deepcopy(result['boxes'])
        primary=result['primary_index']
        assert primary is None if not boxes else primary is not None and 0<=primary<len(boxes)
        features=[]
        crops=[]
        for box in boxes:
            index=box['species_info']['feature_index']
            expected=square_context(image,box['xyxy'])
            actual=result['_crops'][index]
            if actual.size!=expected.size or actual.tobytes()!=expected.tobytes():
                raise ValueError('Original worker crop differs from square_context')
            crops.append(expected)
            features.append(np.asarray(result['_base_features'][index],dtype=np.float32))
        bio=np.stack(features) if features else np.empty((0,768),dtype=np.float32)
        prefix=bio_prefix(worker.base.encoder,crops,device) if crops and need_bio_prefix else None
        metadata={key:result[key] for key in ('detection_mode','normal_prefix_length','normal_output_preserved',
                    'detection_origins','ordinary_prefix_length','total_ms','numeric_mode','device_label') if key in result}
        metadata['detector_configuration']=deepcopy(worker.config['detector'])
        metadata['detector_asset_sha256']={key:value for key,value in worker.config['asset_hashes'].items()
                                           if key.startswith('models/detection/')}
        return boxes,primary,crops,bio,prefix,metadata
    finally:
        worker.close()
        del worker
        release(device)


def encode_bio25(crops,device):
    from encoders import bioclip
    encoder=bioclip(device,'bioclip25',4)
    try:
        features=encoder.encode_images(crops)[0]
        assert features.shape==(len(crops),1024) and np.isfinite(features).all()
        return features
    finally:
        encoder.close()
        del encoder
        release(device)


def encode_dino(crops,device,architecture):
    """Keep DINO's own official transform separate from OpenCLIP's transform."""
    from encoders import DinoEncoder
    encoder=DinoEncoder(device)
    try:
        assert len(encoder.model.blocks)==12 and not encoder.model.chunked_blocks
        assert encoder.model.num_register_tokens==0
        pieces=[]
        if architecture=='bioclip_tail':
            # Bio-tail training uses original frozen DINO CLS features (batch<=8).
            for begin in range(0,len(crops),8):
                pieces.append(encoder.encode(crops[begin:begin+8]))
            features=np.concatenate(pieces).astype(np.float32,copy=False)
            assert features.shape==(len(crops),384) and np.isfinite(features).all()
            return None,features
        # DINO-tail consumes FP32 residual tokens after exactly the first 11 blocks.
        for begin in range(0,len(crops),16):
            batch=torch.stack([encoder.preprocess(image.convert('RGB')) for image in crops[begin:begin+16]]).to(device)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
                tokens=encoder.model.prepare_tokens_with_masks(batch)
                for block in encoder.model.blocks[:-1]:
                    tokens=block(tokens)
                assert tuple(tokens.shape[1:])==(257,384)
                assert tokens.dtype==torch.float32 and torch.isfinite(tokens).all()
            pieces.append(tokens.float().cpu().numpy())
        tokens=np.concatenate(pieces).astype(np.float32,copy=False)
        return tokens,None
    finally:
        encoder.model=None
        del encoder
        release(device)


def species_probabilities(spec,crops,bio,bio_tokens,device):
    from scoring import family_logits, BASELINE
    from evaluate import raw_base, apply_residual
    started=time.perf_counter()
    base=None
    if spec['form']=='full_residual':
        bio25=encode_bio25(crops,device)
        base=raw_base(bio,bio25)[BASELINE]
        del bio25
    dino_tokens,dino=encode_dino(crops,device,spec['architecture'])
    stage={'rows':[{} for _ in crops],'present':np.ones(len(crops),dtype=bool),
           'bio':bio,'bio_tokens':bio_tokens,'tokens':dino_tokens,'dino':dino}
    individual,timing=family_logits(spec['models'],{'photo':stage},device,32)
    logits=np.mean(individual['photo'],axis=0)
    assert logits.shape==(len(crops),len(CLASSES)) and np.isfinite(logits).all()
    if spec['form']=='closed':
        probabilities=softmax(logits/spec['temperature'],axis=1)
        classes=CLASSES
    else:
        probabilities=apply_residual(base,logits,spec['alpha'],spec['temperature'])
        classes=np.arange(len(LABELS),dtype=np.int64)
    assert probabilities.shape==(len(crops),len(classes))
    assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1),1.)
    return probabilities,classes,{'species_pipeline_ms':(time.perf_counter()-started)*1000,
                                   'tail_ensemble_timings':timing}


def export(image,source,folder,selection,spec,selection_hash,boxes,primary,crops,probabilities,classes,metadata,timing):
    annotated=image.copy()
    draw=ImageDraw.Draw(annotated)
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    size=max(16,min(30,image.width//45))
    font=ImageFont.truetype(str(font_path),size) if font_path.exists() else ImageFont.load_default()
    candidates=[]
    for number,(box,patch,p) in enumerate(zip(boxes,crops,probabilities),1):
        before=box['species_info']
        order=np.argsort(-p,kind='stable')[:5]
        identifier=int(classes[order[0]])
        label=LABELS[identifier]
        accepted=bool(before['accepted'])
        filename=f'bird_{number:03d}.png'
        patch.save(folder/filename)
        selected=number-1==primary
        display=(label.get('common_name') or label['scientific_name']) if accepted else '鸟（种类不确定）'
        color='#ed513b' if selected else '#1a9d72' if accepted else '#d99100'
        x1,y1,x2,y2=box['xyxy']
        draw.rectangle(box['xyxy'],outline=color,width=max(2,image.width//400))
        draw.text((x1,max(0,y1-size-6)),f'{number} {display}',font=font,fill=color)
        candidates.append({'candidate_index':number-1,'xyxy':deepcopy(box['xyxy']),
            'detector_confidence':box['confidence'],'original_detection':deepcopy(box),
            'primary':selected,'species_global_id':identifier,'scientific_name':label['scientific_name'],
            'common_name':label.get('common_name'),'display_name':display,
            'species_probability':float(p[order[0]]),'accepted_original_E':accepted,'confirmed':accepted,
            'confirmation_basis':'Unchanged original E gate; no new species-head rejection threshold was introduced',
            'birdness_original_E':before['birdness_score'],'original_species_info':deepcopy(before),
            'top5':[{'global_id':int(classes[i]),'scientific_name':LABELS[int(classes[i])]['scientific_name'],
                     'common_name':LABELS[int(classes[i])].get('common_name'),'probability':float(p[i])} for i in order],
            'crop':filename,'crop_sha256':sha(folder/filename),'crop_size':list(patch.size),
            'crop_rule':'Exact original square_context, factor1.2; PNG preserves classifier input pixels'})
    assert len(candidates)==len(boxes) and [r['xyxy'] for r in candidates]==[b['xyxy'] for b in boxes]
    assert [r['accepted_original_E'] for r in candidates]==[bool(b['species_info']['accepted']) for b in boxes]
    assert [i for i,r in enumerate(candidates) if r['primary']]==([] if primary is None else [primary])
    annotated.save(folder/'annotated.jpg',quality=95)
    summary={'source_image':str(Path(source).resolve()),'source_sha256':sha(source),'image_size':list(image.size),
        'selection_sha256':selection_hash,'candidate_spec':deepcopy(spec),'candidate_id':spec['id'],
        'is_validation_selected_deployment_candidate':selection.get('deployment_candidate_id')==spec['id'],
        'experiment_only':True,'detection_mode':'normal','YOLO_detection_executed':True,
        'original_detector_boxes_and_primary_preserved':True,'original_E_rejection_gate_preserved':True,
        'primary_index':primary,'candidate_count':len(candidates),'count_is_unique_individual_count':False,
        'classification_label_count':len(classes),'classification_global_ids':[int(i) for i in classes],
        'closed_set':spec['form']=='closed','seed_ensemble':'Mean of three seed logits, then frozen validation temperature/residual',
        'sequential_large_backbone_loading':True,'detector':metadata,'timings_ms':timing,
        'candidates':candidates,'annotated_image':'annotated.jpg',
        'numeric_reference':'CUDA BF16 encoder math, FP32 storage/head inputs; CPU mode has not had full accuracy regression',
        'candidate_count_note':'Preserves original duplicate/false-positive boxes; number is candidate crops, not exact bird individuals'}
    save(folder/'result.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True)
    parser.add_argument('--output',required=True,help='New or empty output folder')
    parser.add_argument('--candidate',required=True,help='Exact spec ID from final selection.json candidates')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=parser.parse_args()
    source=Path(args.image).resolve()
    folder=Path(args.output).resolve()
    if not source.is_file():
        parser.error('Image path is not a file.')
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        parser.error('Choose a new or empty output folder; existing outputs will not be overwritten.')
    try:
        selection,spec,selection_hash=select_candidate(args.candidate)
    except (ValueError,AssertionError) as error:
        parser.error(str(error))
    with Image.open(source) as original:
        image=ImageOps.exif_transpose(original).convert('RGB')
    torch.set_num_threads(4)
    folder.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    boxes,primary,crops,bio,bio_tokens,detector=normal_detection(image,args.device,spec['architecture']=='bioclip_tail')
    timing={'detection_and_original_E_ms':(time.perf_counter()-started)*1000}
    classes=CLASSES if spec['form']=='closed' else np.arange(len(LABELS),dtype=np.int64)
    probabilities=np.empty((0,len(classes)),dtype=np.float64)
    if crops:
        probabilities,classes,species_timing=species_probabilities(spec,crops,bio,bio_tokens,args.device)
        timing.update(species_timing)
    timing['total_ms']=(time.perf_counter()-started)*1000
    if sha(HERE/'selection.json')!=selection_hash:
        raise RuntimeError('Frozen selection changed during inference; refusing to export.')
    result=export(image,source,folder,selection,spec,selection_hash,boxes,primary,crops,
                  probabilities,classes,detector,timing)
    print('Saved',result['candidate_count'],'candidate crops to',folder,flush=True)


if __name__=='__main__':
    main()
