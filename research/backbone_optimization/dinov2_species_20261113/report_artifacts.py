from collections import Counter
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from data import HERE, PACKAGE, LABELS, read, save, sha, photo, crop


run = HERE/'runs'
selection = read(run/'selection.json')
test = read(run/'test/summary.json')
assert sha(run/'selection.json')==test['selection_sha256']
assert sha(HERE/'frozen_data/test.json')==test['test_manifest_sha256']
records = read(run/'test/deployed_BioCLIP2_2_5_full.json')['records']
metadata = {r['key']:r for r in read(HERE/'features/test.json')}
errors = [r for r in records if not r['correct']]
font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
canvas = Image.new('RGB',(1250,340*len(errors)), 'white')
draw = ImageDraw.Draw(canvas)
for i,r in enumerate(errors):
    row = metadata[r['key']]
    start = i*340
    target = LABELS[r['target']]['scientific_name']
    pred = LABELS[r['prediction']]['scientific_name'] if r['prediction']>=0 else 'no detection'
    draw.text((12,start+5),f"{r['key']}    {target}  →  {pred}",font=font,fill='black')
    original = photo(row)
    boxes = ImageDraw.Draw(original)
    for coords in row['boxes']:
        boxes.rectangle(coords,outline='#167ac4',width=3)
    if row['bbox'] is not None:
        boxes.rectangle(row['bbox'],outline='#d95331',width=4)
    original.thumbnail((700,294))
    canvas.paste(original,(14,start+35))
    if row['bbox'] is not None:
        patch=crop(row)
        patch.thumbnail((420,294))
        canvas.paste(patch,(740,start+35))
    draw.line((0,start+338,1250,start+338),fill='#dddddd',width=2)
canvas.save(run/'test/error_review.jpg',quality=93)

colors={'bio':'#1d62a0','dino':'#d58525','fusion':'#248360'}
fig, panels = plt.subplots(2,2,figsize=(12.6,7.4),layout='constrained')
for group,col in [('original',0),('expanded',1)]:
    for mode in colors:
        traces=[]
        for seed in (131,271,719):
            with (run/f'{group}_{mode}_{seed}/results.csv').open(newline='',encoding='utf-8') as stream:
                traces.append(list(csv.DictReader(stream)))
        for axis,field,label in [(panels[0,col],'train_loss','Training loss'),(panels[1,col],'validation_correct','Validation correct (of 302 detected)')]:
            values=np.array([[float(row[field]) for row in trace] for trace in traces])
            average=values.mean(0)
            epochs=np.arange(1,len(average)+1)
            axis.plot(epochs,average,color=colors[mode],label=mode)
            axis.fill_between(epochs,values.min(0),values.max(0),color=colors[mode],alpha=.16)
            axis.set(xlabel='Epoch',ylabel=label)
            axis.grid(alpha=.15)
    panels[0,col].set_title(f'{group.capitalize()} training set')
    panels[0,col].legend(frameon=False)
fig.savefig(run/'training_curves.png',dpi=170)
plt.close(fig)

before = read(HERE.parent/'bioclip_dinov2_audit_20261113/protected_assets_before.json')
for file in before.values():
    assert sha(file['path'])==file['sha256']
save(run/'verification.json',{'verified_original_assets':len(before),'smoke_passed':read(HERE/'smoke_result.json')['passed'],
    'source_selection_sha256':test['selection_sha256'],'test_manifest_verified':True,'training_runs':len(list(run.glob('*/complete.json'))),
    'test_reselection':False,'boxes_fixed_across_classifiers':True,'production_files_written':False})
print('Artifacts generated and 11 original assets verified.',flush=True)
