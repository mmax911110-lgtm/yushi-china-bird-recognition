"""Apply a frozen experimental head to a single pre-cropped RGB bird image."""
import argparse
import json
from PIL import Image, ImageOps
import numpy as np
from scipy.special import softmax
import torch
from data import HERE, CLASSES, LABELS, read
from encoders import DinoEncoder, bioclip
from evaluate import raw_base, apply_residual, cached_logits


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--image', required=True, help='One already cropped bird image')
parser.add_argument('--candidate', required=True, help='Candidate ID in runs/selection.json')
parser.add_argument('--device', choices=('cuda','cpu'), default='cuda')
args = parser.parse_args()
selection = read(HERE/'runs/selection.json')
matches = [c for c in selection['candidates'] if c['id']==args.candidate]
if len(matches)!=1:
    parser.error('Use a frozen candidate ID: '+', '.join(c['id'] for c in selection['candidates']))
spec = matches[0]
with Image.open(args.image) as original:
    image = ImageOps.exif_transpose(original).convert('RGB')
x = np.zeros((1,768),dtype=np.float32)
y = np.zeros((1,384),dtype=np.float32)
if spec['mode'] != 'dino' or spec['form']=='full_residual':
    encoder = bioclip(args.device)
    x = encoder.encode_images([image])[0]
    encoder.close()
if spec['mode'] != 'bio':
    encoder = DinoEncoder(args.device)
    y = encoder.encode([image])
    del encoder
logits = np.mean(cached_logits(spec['models'],x,y),axis=0)
if spec['form']=='closed':
    probabilities = softmax(logits/spec['temperature'],axis=1)[0]
    classes = CLASSES
else:
    encoder = bioclip(args.device,'bioclip25',batch=2)
    x25 = encoder.encode_images([image])[0]
    encoder.close()
    base = raw_base(x,x25)['deployed_BioCLIP2_2_5']
    probabilities = apply_residual(base,logits,spec['alpha'],spec['temperature'])[0]
    classes = np.arange(len(LABELS))
order = np.argsort(-probabilities,kind='stable')[:5]
print(json.dumps({'candidate':spec['id'],'species':len(classes),'already_cropped_input':True,
    'predictions':[{'global_id':int(classes[i]),'scientific_name':LABELS[classes[i]]['scientific_name'],
    'name':LABELS[classes[i]].get('common_name'),'score':float(probabilities[i])} for i in order]},ensure_ascii=True,indent=2))
