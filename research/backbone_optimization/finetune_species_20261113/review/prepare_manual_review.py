"""Create CPU-only complete error/change review cards from frozen test results."""
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import math
import textwrap
from PIL import Image,ImageDraw,ImageFont,ImageOps

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
PACKAGE=Path(r'D:\项目\鸟类识别\识别模型\v1\v1.1\NVIDIA')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def fixed_crop(image,coords):
    # Exact arithmetic and padding from protected engine_n.square_context, no model import.
    x1,y1,x2,y2=map(float,coords)
    assert all(map(math.isfinite,(x1,y1,x2,y2))) and x2>x1 and y2>y1
    side=max(2,int(math.ceil(max(x2-x1,y2-y1)*1.2)))
    left=int(math.floor((x1+x2-side)/2));top=int(math.floor((y1+y2-side)/2))
    bounds=(max(0,left),max(0,top),min(image.width,left+side),min(image.height,top+side))
    assert bounds[2]>bounds[0] and bounds[3]>bounds[1]
    square=Image.new('RGB',(side,side),(114,114,114))
    square.paste(image.crop(bounds),(bounds[0]-left,bounds[1]-top))
    return square,{'factor':1.2,'side':side,'left':left,'top':top,'source_bounds':bounds}


def main():
    selection_path=EXP/'selection.json'
    summary_path=EXP/'runs/test/summary.json'
    report_path=EXP/'report_candidate.json'
    summary,report=read(summary_path),read(report_path)
    assert summary['selection_sha256']==sha(selection_path)==report['selection_sha256']
    assert report['choice_made_before_test_features'] and report['deployment_candidate_id'] is None
    assert read(selection_path)['deployment_candidate_id'] is None
    source=read(EXP/'features/test.json')
    assert len(source)==95
    keyed={row['key']:row for row in source}
    labels=read(PACKAGE/'configs/china_labels.json')
    names={i:row.get('common_name') or row['scientific_name'] for i,row in enumerate(labels)}
    names[-1]='无框 / no crop'
    outputs={name:read(EXP/'runs/test'/(name+'.json')) for name in summary['metrics']}
    records={name:{row['key']:row for row in value['records']} for name,value in outputs.items()}
    baselines={1343:records['deployed_BioCLIP2_2_5_full'],80:records['deployed_BioCLIP2_2_5_closed']}
    changes=defaultdict(list)
    for model,rows in records.items():
        vocab=summary['metrics'][model]['class_count']
        for key,row in rows.items():
            base=baselines[vocab][key]
            if row['prediction']!=base['prediction']:
                changes[key].append({'model':model,'vocabulary':vocab,'baseline_prediction':base['prediction'],
                    'baseline_name':names[base['prediction']],'prediction':row['prediction'],'name':names[row['prediction']],
                    'baseline_correct':base['correct'],'correct':row['correct']})
    report_errors={row['key'] for row in outputs[report['id']]['records'] if not row['correct']}
    required=set(changes)|report_errors
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    title=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    card_folder=HERE/'cards';crop_folder=HERE/'fixed_crops'
    card_folder.mkdir(exist_ok=True);crop_folder.mkdir(exist_ok=True)
    cases=[]
    for row in source:
        key=row['key']
        if key not in required:
            continue
        ordinal=len(cases)+1
        with Image.open(row['path']) as original:
            image=ImageOps.exif_transpose(original).convert('RGB')
        assert sha(row['path'])==row['sha256']
        canvas=Image.new('RGB',(1200,1050),(246,246,246))
        draw=ImageDraw.Draw(canvas)
        base_full=baselines[1343][key]
        base_closed=baselines[80][key]
        proposed=records[report['id']][key]
        draw.text((12,8),f'{ordinal:02d} {key}   Source label: {names[row["target"]]}',font=title,fill='black')
        draw.text((12,42),f'Old full: {names[base_full["prediction"]]} | Old closed: {names[base_closed["prediction"]]} | Report: {names[proposed["prediction"]]}',font=font,fill='black')
        display=ImageOps.contain(image,(752,610))
        left=(772-display.width)//2;top=92+(610-display.height)//2
        canvas.paste(display,(left,top))
        crop_path=None;crop_geometry=None
        if row['bbox'] is not None:
            x1,y1,x2,y2=row['bbox'];sx=display.width/image.width;sy=display.height/image.height
            draw.rectangle((left+x1*sx,top+y1*sy,left+x2*sx,top+y2*sy),outline='red',width=3)
            crop,crop_geometry=fixed_crop(image,row['bbox'])
            crop_path=crop_folder/f'{ordinal:02d}_{key}.png'
            crop.save(crop_path)
            crop_display=ImageOps.contain(crop,(400,400))
            if max(crop_display.size)<400:
                scale=400/max(crop_display.size)
                crop_display=crop_display.resize((round(crop_display.width*scale),round(crop_display.height*scale)),Image.Resampling.NEAREST)
            canvas.paste(crop_display,(790+(400-crop_display.width)//2,150+(400-crop_display.height)//2))
            draw.text((790,580),f'Actual fixed 1.2 crop: {crop.width}x{crop.height}',font=small,fill='black')
        else:
            draw.text((800,280),'NO BBOX',font=title,fill='red')
            draw.text((800,320),'No classifier crop exists.',font=small,fill='black')
        draw.text((790,94),'Unmodified original pixels in crop',font=small,fill='black')
        draw.text((12,718),f'Original {image.width}x{image.height}; accepted={row["accepted"]}; detector boxes={len(row.get("detector_boxes",[]))}',font=small,fill='black')
        draw.text((12,744),'Manual review only: keep source labels, fixed crop, policy and all 95 inputs.',font=small,fill='black')
        lines=[f'Report candidate error: {key in report_errors}; changed predictions across models: {len(changes[key])}']
        grouped=defaultdict(list)
        for change in changes[key]:
            grouped[(change['vocabulary'],change['name'],change['baseline_name'])].append(change['model'])
        for (vocabulary,predicted,previous),models in grouped.items():
            lines.append(f'{vocabulary} classes: {previous} -> {predicted}: '+', '.join(models))
        y=782
        for line in lines:
            for part in textwrap.wrap(line,width=115,break_long_words=False,break_on_hyphens=False):
                draw.text((12,y),part,font=small,fill='black');y+=25
        card_path=card_folder/f'{ordinal:02d}_{key}.jpg'
        canvas.save(card_path,quality=94)
        cases.append({'review_id':ordinal,'key':key,'path':row['path'],'source_sha256':row['sha256'],
            'target':row['target'],'source_name':names[row['target']],
            'scientific_name':labels[row['target']]['scientific_name'],
            'bbox':row['bbox'],'accepted':row['accepted'],'detector_boxes':row.get('detector_boxes'),
            'baseline_full':base_full,'baseline_closed':base_closed,'report_candidate':proposed,
            'all_prediction_changes':changes[key],'report_candidate_error':key in report_errors,
            'fixed_crop_path':None if crop_path is None else str(crop_path),
            'fixed_crop_sha256':None if crop_path is None else sha(crop_path),'fixed_crop_geometry':crop_geometry,
            'card_path':str(card_path),'card_sha256':sha(card_path)})
    active_changes={key:value for key,value in changes.items() if value}
    result={'status':'awaiting_manual_visual_review','created_utc':datetime.now(timezone.utc).isoformat(),
        'selection_sha256':sha(selection_path),'test_summary_sha256':sha(summary_path),
        'report_candidate_sha256':sha(report_path),'report_candidate_summary':report,
        'source_rows_sha256':sha(EXP/'features/test.json'),'primary_denominator':95,
        'report_candidate_error_keys':sorted(report_errors),'changed_prediction_keys':sorted(active_changes),
        'required_union_keys':sorted(required),'cases':cases,
        'model_used_to_change_labels':False,'source_labels_changed':False,'denominator_changed':False}
    assert {case['key'] for case in cases}==required==set(active_changes)|report_errors
    (HERE/'manual_review_inputs.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'cases':len(cases),'report_errors':len(report_errors),'changed_prediction_images':len(active_changes),
                      'no_box':sum(case['bbox'] is None for case in cases),
                      'case_index':[{k:case[k] for k in ('review_id','key','source_name','card_path')} for case in cases]},ensure_ascii=True))


if __name__=='__main__':
    main()
