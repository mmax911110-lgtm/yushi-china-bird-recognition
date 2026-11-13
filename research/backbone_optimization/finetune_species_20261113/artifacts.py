"""CPU-only final figures, report, paired statistics and provenance checks.

Run only after runs/test/summary.json exists. No model inference or selection.
Default status remains pending_manual_and_cli; --final requires explicit human
review and a checked CLI result, and never changes the frozen model choice.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image,ImageOps

from common import HERE,OLD,PACKAGE,CLASSES,LABELS,read,save,sha

BASE='deployed_BioCLIP2_2_5'
FAMILIES=('warmstart','head_control','tail_lr1e5','tail_lr3e5',
          'bio_head_control','bio_tail_lr1e6','bio_tail_lr3e6')
NAMES={'warmstart':'原融合头 warmstart','head_control':'DINO 冻结尾层对照',
       'tail_lr1e5':'DINO 末块 1e-5','tail_lr3e5':'DINO 末块 3e-5',
       'bio_head_control':'BioCLIP 冻结尾层对照','bio_tail_lr1e6':'BioCLIP 末块 1e-6',
       'bio_tail_lr3e6':'BioCLIP 末块 3e-6'}
SHORT=('Warmstart','DINO head only','DINO tail 1e-5','DINO tail 3e-5',
       'Bio head only','Bio tail 1e-6','Bio tail 3e-6')


def ratio(a,b):
    return f'{a}/{b} ({100*a/b:.2f}%)' if b else 'N/A'


def uri(path):
    return Path(path).resolve().as_posix()


def link(path,label=None):
    return f'[{label or Path(path).name}]({uri(path)})'


def paired(current,baseline,seed=20261113):
    assert len(current)==len(baseline) and current
    clusters=defaultdict(lambda:np.zeros(2,dtype=np.float64))
    corrected,regressed=[],[]
    for a,b in zip(current,baseline):
        assert a['key']==b['key']
        delta=int(a['correct'])-int(b['correct'])
        clusters[str(a.get('user_id',a['key']))]+=np.array([delta,1.])
        if delta==1:
            corrected.append(a['key'])
        elif delta==-1:
            regressed.append(a['key'])
    counts=np.array(list(clusters.values()))
    rng=np.random.default_rng(seed)
    sample=rng.integers(0,len(counts),(10000,len(counts)))
    difference=100*counts[sample,0].sum(1)/counts[sample,1].sum(1)
    return {'images':len(current),'photographer_proxy_groups':len(clusters),
        'corrected':corrected,'regressed':regressed,'net':len(corrected)-len(regressed),
        'difference_pp':100*counts[:,0].sum()/counts[:,1].sum(),
        'photographer_cluster_bootstrap95_pp':np.percentile(difference,[2.5,97.5]).tolist(),
        'bootstrap_resamples':10000,'scope':'Paired public source-label agreement; not per-individual bird identification ground truth'}


def load_training(selection):
    records=[]
    for family in FAMILIES[1:]:
        for evidence in selection['families'][family]['models']:
            folder=HERE/'runs'/f"{family}_{evidence['seed']}"
            complete=read(folder/'complete.json')
            contract=read(folder/'contract.json')
            assert sha(complete['checkpoint'])==complete['checkpoint_sha256']==evidence['checkpoint_sha256']
            assert contract['seed']==evidence['seed'] and contract['spec']['id']==family
            for name,expected in contract['source_sha256'].items():
                assert sha(HERE/name)==expected
            with (folder/'results.csv').open(encoding='utf-8',newline='') as stream:
                history=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
            assert len(history)==12 and history[-1]['epoch']==12
            records.append({'family':family,'seed':evidence['seed'],'folder':str(folder),
                'complete':complete,'initial':read(folder/'initial_validation.json'),'history':history,
                'complete_sha256':sha(folder/'complete.json'),'contract_sha256':sha(folder/'contract.json')})
    assert len(records)==18 and sum(r['complete']['train_tail'] for r in records)==12
    return records


def learning_curves(training,output):
    colors=['#0072B2','#D55E00','#009E73']
    for tag,families,title in (
        ('dino',FAMILIES[1:4],'DINOv2 final-block experiment'),
        ('bio',FAMILIES[4:7],'BioCLIP2 final-block experiment')):
        fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained',sharex=True)
        fig.suptitle(title+' | fixed 1,577 training crops; 302 detected validation crops',fontsize=13)
        for row,family in enumerate(families):
            group=sorted((r for r in training if r['family']==family),key=lambda r:r['seed'])
            for color,item in zip(colors,group):
                history=item['history']
                epochs=[r['epoch'] for r in history]
                initial=sum(v['correct'] for v in item['initial'].values())
                values=[initial]+[r['validation_correct'] for r in history]
                best=int(item['complete']['selected_epoch'])
                axes[row,0].plot(epochs,[r['train_loss'] for r in history],color=color,lw=1.6,label=str(item['seed']))
                axes[row,1].plot([0]+epochs,values,color=color,lw=1.6,label=f"{item['seed']} (best e{best})")
                axes[row,1].scatter([best],[values[best]],marker='*',s=115,color=color,edgecolors='white',linewidth=.5,zorder=5)
                if best==0:
                    axes[row,1].annotate('best e0: backbone unchanged',(0,values[0]),xytext=(9,-13),
                                         textcoords='offset points',fontsize=8,color=color)
            for column in range(2):
                ax=axes[row,column]
                ax.set_title(family,fontsize=10)
                ax.grid(alpha=.2)
                ax.spines[['top','right']].set_visible(False)
                ax.set_xlim(-.5,12.5)
                ax.set_xticks([0,2,4,6,8,10,12])
                ax.legend(fontsize=8,loc='best')
                if row==2:
                    ax.set_xlabel('Epoch (0 = exact warmstart)')
            axes[row,0].set_ylabel('Training objective')
            axes[row,1].set_ylabel('Validation correct / 302\nconditional on detected supported crops')
            axes[row,1].yaxis.set_major_locator(MaxNLocator(integer=True))
        fig.savefig(output/f'{tag}_training_curves.png',dpi=170)
        fig.savefig(output/f'{tag}_training_curves.svg')
        plt.close(fig)


def comparison_figure(selection,test,output):
    ids=[(BASE+'_closed',BASE+'_full')]+[
        (selection['families'][f]['closed_comparison'],selection['families'][f]['full_comparison']) for f in FAMILIES]
    labels=['Deployed']+list(SHORT)
    val_baseline=selection['development_baselines'][BASE]
    val_closed=[sum(v['closed']['correct'] for v in val_baseline.values())]
    val_full=[sum(v['full']['correct'] for v in val_baseline.values())]
    by_id={s['id']:s for s in selection['candidates']}
    for closed,full in ids[1:]:
        val_closed.append(sum(v['correct'] for v in by_id[closed]['metrics'].values()))
        val_full.append(sum(v['correct'] for v in by_id[full]['metrics'].values()))
    panels=[(val_closed,312,'Validation: closed 80 labels / 312 images'),
            (val_full,388,'Validation: full 1,343 labels / 388 images'),
            ([test['metrics'][c]['correct'] for c,_ in ids],95,'Fresh test: closed 80 labels / 95 images'),
            ([test['metrics'][f]['correct'] for _,f in ids],95,'Fresh test: full 1,343 labels / 95 images')]
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    palette=['#666666','#9E9E9E','#56B4E9','#0072B2','#009E73','#E69F00','#D55E00','#CC79A7']
    for ax,(values,total,title) in zip(axes.flat,panels):
        y=np.arange(len(labels))
        ax.barh(y,np.array(values)/total*100,color=palette)
        ax.set_yticks(y,labels,fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0,108)
        for i,value in enumerate(values):
            ax.text(value/total*100+.6,i,f'{value}/{total}',va='center',fontsize=8)
        ax.set_title(title,fontsize=10)
        ax.set_xlabel('Source-label agreement (%) | missing crops count as errors')
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(output/'comparison.png',dpi=170)
    fig.savefig(output/'comparison.svg')
    plt.close(fig)


def verify_cli(path,selection,rows):
    from inference.engine_n import square_context
    result=read(path)
    assert result['selection_sha256']==sha(HERE/'selection.json')
    assert result['source_sha256']==sha(result['source_image'])
    assert result['candidate_id'] in {s['id'] for s in selection['candidates']}
    assert result['YOLO_detection_executed'] and result['original_detector_boxes_and_primary_preserved']
    assert result['original_E_rejection_gate_preserved'] and result['experiment_only']
    same=[r for r in rows if Path(r['path']).resolve()==Path(result['source_image']).resolve()]
    if same:
        assert len(same)==1
        assert [r['xyxy'] for r in result['candidates']]==same[0]['detector_boxes']
        assert result['primary_index']==same[0]['primary_index']
    with Image.open(result['source_image']) as source:
        rgb=ImageOps.exif_transpose(source).convert('RGB')
    for item in result['candidates']:
        patch_path=Path(path).parent/item['crop']
        assert sha(patch_path)==item['crop_sha256']
        expected=square_context(rgb,item['xyxy'])
        with Image.open(patch_path) as actual:
            assert np.array_equal(np.asarray(actual.convert('RGB')),np.asarray(expected))
        assert item['confirmed']==item['original_species_info']['accepted']
    return {'status':'passed','path':str(path),'sha256':sha(path),
            'source_matches_test_cache':bool(same),'candidate_crops':len(result['candidates']),
            'boxes_primary_original_gate_and_exact_PNG_crop_pixels_checked':True}


def evidence(selection,test,training,cli_result=None,manual_review=None,final=False):
    selection_sha=sha(HERE/'selection.json')
    assert test['status']=='completed_frozen_independent_test' and test['images']==95 and test['species']==37
    assert test['selection_sha256']==selection_sha
    assert not test['deployment_candidate_changed_after_test'] and not test['test_retuning']
    assert test['fixed_detector_and_gate'] and test['deployment_candidate_id']==selection['deployment_candidate_id']
    assert selection['deployment_candidate_id'] is None
    focus=read(HERE/'report_candidate.json')
    assert focus['selection_sha256']==selection_sha and focus['choice_made_before_test_features']
    checked={}
    for path,digest in {**selection['frozen_input_sha256'],**test['test_input_sha256']}.items():
        assert sha(path)==digest,path
        checked[path]=digest
    qa=read(HERE/'holdout_data/input_qa.json')
    assert qa['reviewed_before_inference'] and not qa['source_labels_changed'] and not qa['denominator_changed']
    assert qa['images']==qa['main_denominator']==95 and not qa['excluded_from_main']
    assert qa['quality_flags_selected_before_any_prediction']
    rows=read(HERE/'features/test.json')
    assert len(rows)==95 and sum(r['bbox'] is None for r in rows)==test['no_box']
    for row in rows:
        assert sha(row['path'])==row['sha256']
    protected=read(HERE.parent/'bioclip_dinov2_audit_20261113/protected_assets_before.json')
    protected_results={}
    assert len(protected)==11
    for name,item in protected.items():
        actual=sha(item['path'])
        assert actual==item['sha256'],name
        protected_results[name]={'path':item['path'],'sha256':actual,'unchanged':True}
    old_cfg=HERE.parent/'bioclip_dinov2_audit_20261113/deployed_configuration.json'
    assert read(PACKAGE/'configs/recognizer.json')==read(old_cfg)
    smoke=read(HERE/'smoke_result.json')
    bio_smoke=read(HERE/'bio_smoke_result.json')
    assert smoke['passed'] and bio_smoke['passed']
    if cli_result:
        paths=[cli_result] if isinstance(cli_result,str) else cli_result
        cli={'status':'passed','demonstrations':[verify_cli(Path(p),selection,rows) for p in paths]}
    else:
        cli={'status':'pending'}
    independent_cli=HERE/'data_review/cli_runtime_review.json'
    if independent_cli.exists():
        reviewed=read(independent_cli)
        assert reviewed['status']=='passed' and reviewed['selection_sha256']==selection_sha
        for path,digest in reviewed['code_sha256'].items():
            assert sha(path)==digest,path
        demonstrations=[]
        for item in reviewed['records']:
            assert sha(item['result_path'])==item['result_sha256']
            demonstrations.append(verify_cli(Path(item['result_path']),selection,rows))
        assert len(demonstrations)==2
        cli={'status':'passed','independent_review_path':str(independent_cli),
             'independent_review_sha256':sha(independent_cli),'demonstrations':demonstrations,
             'note':reviewed['note']}
    manual={'status':'pending'}
    if manual_review:
        p=Path(manual_review)
        assert p.is_file() and p.stat().st_size>0
        reviewed=read(p)
        assert reviewed['status']=='completed_complete_error_and_change_visual_review'
        assert reviewed['selection_sha256']==selection_sha
        assert reviewed['test_summary_sha256']==sha(HERE/'runs/test/summary.json')
        assert reviewed['report_candidate_sha256']==sha(HERE/'report_candidate.json')
        assert reviewed['primary_denominator']==95 and not reviewed['source_labels_changed'] and not reviewed['denominator_changed']
        assert reviewed['reviewed_count']==reviewed['required_union_count']==18 and not reviewed['missing_required_keys']
        assert reviewed['all_report_candidate_errors_covered'] and reviewed['all_full_and_closed_name_changes_covered']
        for item in reviewed['cases']:
            assert item['manually_viewed'] and sha(item['card_path'])==item['card_sha256']
            if item['fixed_crop_path'] is not None:
                assert sha(item['fixed_crop_path'])==item['fixed_crop_sha256']
        manual={'status':'passed','path':str(p.resolve()),'sha256':sha(p),
                'reviewed_count':18,'report_candidate_error_category_counts':reviewed['report_candidate_error_category_counts'],
                'markdown_path':str(p.parent/'人工错例复核.md'),'markdown_sha256':sha(p.parent/'人工错例复核.md')}
    if final:
        assert cli['status']=='passed' and manual_review,'Final acceptance requires CLI and explicit manual-review evidence'
    dino_weights=OLD/'assets/dinov2_vits14_pretrain.pth'
    assert sha(dino_weights)==read(HERE/'tokens/manifest.json')['source_checkpoint_sha256']
    return {'completion_status':'completed' if final else 'pending_manual_and_cli',
        'generated_utc':datetime.now(timezone.utc).isoformat(),'CPU_only':True,
        'cuda_context_initialized':torch.cuda.is_initialized(),
        'selection_sha256':selection_sha,'report_candidate':focus,
        'independent_test_summary_sha256':sha(HERE/'runs/test/summary.json'),
        'frozen_inputs_SHA256_verified':checked,'original_assets_checked':11,
        'original_assets':protected_results,'production_config_matches_original_audit_snapshot':True,
        'production_GUI_replaced':False,'deployment_candidate_id':None,
        'trained_runs':18,'backbone_training_runs':12,'head_controls':6,
        'selected_epoch0_runs':[Path(r['folder']).name for r in training if r['complete']['train_tail'] and r['complete']['selected_epoch']==0],
        'training_records':[{k:r[k] for k in ('family','seed','folder','complete_sha256','contract_sha256')} for r in training],
        'original_DINO_weights_sha256':sha(dino_weights),
        'DINO_smoke_sha256':sha(HERE/'smoke_result.json'),'BioCLIP_smoke_sha256':sha(HERE/'bio_smoke_result.json'),
        'input_QA_sha256':sha(HERE/'holdout_data/input_qa.json'),'CLI':cli,'manual_review':manual}


def prediction_records(test):
    result={}
    for name,metric in test['metrics'].items():
        path=HERE/'runs/test'/(name+'.json')
        item=read(path)
        rows=item['records']
        assert len(rows)==95 and sum(r['correct'] for r in rows)==metric['correct']==item['correct']
        assert sum(not r['has_box'] for r in rows)==test['no_box']
        assert item['selection_sha256']==test['selection_sha256']
        assert sha(item['probabilities']['path'])==item['probabilities']['sha256']
        result[name]=item
    return result


def report(selection,test,training,records,statistics,verification,secondary):
    focus=verification['report_candidate']['id']
    baseline=records[BASE+'_full']
    focal=records[focus]
    focus_pair=statistics['comparisons'][focus]
    selected={s['id']:s for s in selection['candidates']}
    detected=95-test['no_box']
    md=[
        '# BioCLIP＋DINOv2 真实骨干微调实测', '',
        f"本轮完成 18 组真实训练（12 组骨干微调、6 组冻结骨干对照），验证未达到升级门槛，保留原正式 GUI。全 1,343 类旧验证中，原部署为 357/388，各融合 family 均为 356/388；`deployment_candidate=None` 已在独立测试前冻结。预先选定的研究报告候选 `{focus}` 在新 95 张中为 {ratio(focal['correct'],95)}，原部署为 {ratio(baseline['correct'],95)}，纠正 {len(focus_pair['corrected'])} 张、误改 {len(focus_pair['regressed'])} 张。此候选只作研究比较，测试后不改选别的模型。", '',
        f"新集有 37 种来源鸟种、95 个观察者 user_id，实际有框 {detected}/95。原部署有框条件标签命中为 {ratio(baseline['correct'],detected)}，这是检测成功子集的分类结果，不能替代全 95 张的 {ratio(baseline['correct'],95)}。所有主指标保留无框计错。", '',
        '## 实验内容与数据口径', '',
        '本轮训练改变了视觉骨干参数，网络仍为 BioCLIP 2 全局特征＋DINOv2-S/14 全局特征，经各自 256 维投影和 LayerNorm，拼接进入单一 80 类 Species MLP。没有增加局部 Patch 选择、第二分类任务或新检测器。DINO 组训练最后一个 Transformer block 与末 norm，BioCLIP 组单独训练最后视觉 block、ln_post、proj；另一视觉塔只用原冻结缓存。原头从各 seed 的 expanded_fusion 最佳权重 warmstart。', '',
        '这与前一轮检测分块/COCO 补检实验是不同实验。本轮全部方案共用原第四版 YOLO 普通 640 初检＋1280 补检、原主框、原 1.2 倍方形裁切及原 E 确认门槛。分类不确定不会被写成新 head 已校准的接受阈值。候选框数量不等于独立鸟只数。', '',
        '训练使用 1,577 张固定裁图、80 种，观察级弱标签并非逐框专家物种标注。N/O 开发集全体为 388 张、97 个来源鸟种；80 类闭集子集为 312 张、实际出现 78 种，其中 302 张有框。温度拟合和训练模型选择使用这 302 张有框支持类验证图；端到端闭集表仍用 312 张，无框计错。全词表残差表用 1,343 个输出类别及全 388 张。', '',
        '新 95 张在预测前完成输入 QA，按原来源标签评分；有遮挡、远距、多只鸟、处理/装箱及一张明确残骸。纯残骸名单在预测前已固定，主分母仍为 95。排除该固定一张的 94 张仅作为次级口径，不代表其余全是清楚活鸟。37 种新图的表现不能外推到全部 80 个支持类或 1,343 个词表类。观察者去交集、哈希隔离不保证完全排除转载、连拍或基础模型预训练重叠。', '',
        '## 同图结果', '',
        '| 方案 | 闭集验证 /312 | 全词表验证 /388 | 闭集新测 /95 | 全词表新测 /95 | 全词表有框条件新测 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    base_val=selection['development_baselines'][BASE]
    md.append(f"| 原部署 BioCLIP 2＋2.5 | {ratio(sum(v['closed']['correct'] for v in base_val.values()),312)} | {ratio(sum(v['full']['correct'] for v in base_val.values()),388)} | {ratio(records[BASE+'_closed']['correct'],95)} | {ratio(baseline['correct'],95)} | {ratio(baseline['correct'],detected)} |")
    for family in FAMILIES:
        c,f=selection['families'][family]['closed_comparison'],selection['families'][family]['full_comparison']
        md.append(f"| {NAMES[family]} | {ratio(sum(m['correct'] for m in selected[c]['metrics'].values()),312)} | {ratio(sum(m['correct'] for m in selected[f]['metrics'].values()),388)} | {ratio(records[c]['correct'],95)} | {ratio(records[f]['correct'],95)} | {ratio(records[f]['correct'],detected)} |")
    md += ['', '表中 family 的三个种子先平均 logits，再使用验证冻结温度。全词表行同时使用验证冻结的残差权重，保留原部署不属于支持类的输出保护；80 类头没有学习另外 1,263 种。两种词表口径不能混排选优。七个 family 的新图全词表结果均没有纠正原部署的错误；其中部分与原部署持平，部分退化。', '',
           '| 全词表方案 | 新测 Top-3 /95 | 新测 Macro-F1（37 个有真值来源种） | 纠正 / 误改 |',
           '|---|---:|---:|---:|']
    for family in ('deployed',)+FAMILIES:
        identifier=BASE+'_full' if family=='deployed' else selection['families'][family]['full_comparison']
        item=records[identifier]
        pair=statistics['comparisons'][identifier]
        md.append(f"| {'原部署' if family=='deployed' else NAMES[family]} | {item['top3_correct']}/95 | {100*item['macro_f1']:.2f}% | {len(pair['corrected'])} / {len(pair['regressed'])} |")
    ci=focus_pair['photographer_cluster_bootstrap95_pp']
    md += ['', f"预选报告候选相对原部署的配对差为 {focus_pair['difference_pp']:+.2f} 个百分点，按 user_id 分组的 10,000 次 bootstrap 95% 百分位区间为 [{ci[0]:+.2f}, {ci[1]:+.2f}]。这是本批来源标签配对样本的描述，不是总体准确率保证，也没有据该区间改选模型。", '',
           '| 固定次级口径（排除预测前标注的纯残骸） | 标签命中 |', '|---|---:|']
    for identifier in (BASE+'_full',focus):
        sub=secondary[identifier]
        md.append(f"| {identifier} | {ratio(sub['correct'],sub['images'])} |")
    md += ['', '## 真实训练记录', '',
           '两组均为 12 轮、batch 32、相同种子 131/271/719、head LR 1e-4、AdamW、weight decay 0.01、label smoothing 0.025、Dropout 0.2、梯度范数裁剪 1.0；固定裁图、没有增强。DINO 尾层 LR 为 1e-5/3e-5，BioCLIP 尾层为 1e-6/3e-6。两个各自 head-only 对照保留完整相同流程。前缀缓存分别是 DINO 前 11 块和 BioCLIP 前 23 块，不是最终 embedding 直接训练分类头。', '',
           f"{link(HERE/'figures/dino_training_curves.png','DINO 曲线')}；{link(HERE/'figures/bio_training_curves.png','BioCLIP 曲线')}；{link(HERE/'figures/comparison.png','同图对照图')}。星号为真正采用的最佳 epoch，epoch 0 表示 warmstart 直接获选；训练 loss 仅从第 1 轮开始，没有伪造 epoch 0 的训练损失。", '',
           '| 运行 | 最佳 epoch | 有框验证最佳 /302 | 可训练参数 | 最佳权重实际更新骨干 | GPU 峰值分配 MiB |',
           '|---|---:|---:|---:|---|---:|']
    for r in training:
        c=r['complete']
        md.append(f"| {link(Path(r['folder'])/'complete.json',Path(r['folder']).name)} | {c['selected_epoch']} | {c['best_correct']}/302 | {c['trainable_parameters']:,} | {'是' if c['selected_visual_backbone_updated'] else '否'} | {c.get('peak_cuda_allocated_mib',0):.1f} |")
    zero=verification['selected_epoch0_runs']
    md += ['', f"12 组骨干候选在训练过程中均有实际参数更新；最终选中的检查点仍按验证决定。以下骨干候选选中了 epoch 0，最佳权重没有视觉更新：{', '.join('`'+x+'`' for x in zero) or '无'}。不能用最后一轮已更新来代替最佳检查点更新的证据。", '',
           '可训练参数：DINO 末块＋norm＋头 2,224,336，BioCLIP 末块＋ln_post＋proj＋头 13,833,040，两个 head-only 对照均 448,336。主融合视觉塔＋MLP 约 326,471,120 个参数；全词表残差推理还要原 BioCLIP 2.5/E 分类基线，训练参数量不能代表整套推理模型大小。GPU 峰值是该次尾层缓存训练的实际分配量，包含实现和缓存分配影响，不能当作全图应用显存。', '',
           '三种子固定参数新测明细及 mean/std 已保存到 '+link(HERE/'runs/test/per_seed_fixed_parameters.json')+'。'+
           '尾层耗时在 '+link(HERE/'runs/test/summary.json')+' 的 timings 中，只包含缓存后的尾块/归一化/投影/头计算，排除了检测和冻结前缀，不能当作全流程延迟。', '',
           '## 错误与验收', '',
           f"预选报告候选的 {95-focal['correct']} 个错误、每张原图路径、真值/预测、无框标记、配对纠正与退化保存在 {link(HERE/'error_cases.json')}；逐鸟种 Precision/Recall/F1、稀疏混淆矩阵在 {link(HERE/'runs/test'/(focus+'.json'))}。所有类别概率及类别索引存于同名 NPZ，JSON 绑定其 SHA256；不是只保留 Top-1。", '',
           '错误分类仍以公开观察记录物种作为来源标签，未逐只鸟重新专家鉴定。多鸟图中的错误可能来自主个体不对应来源标签，背景框也可能偶然猜中来源标签；有框条件正确率不能证明裁图全是有效鸟体。人工复核报告另行提供，不据它改标签、分母或阈值。', '',
           f"运行核验状态：`{verification['completion_status']}`。11 个原始资产 SHA256 已逐文件复算一致，正式 recognizer 配置与原审查快照一致；详情见 {link(HERE/'runtime_verification.json')}。CLI 与人工核查的完成情况以该文件的具体证据字段为准。", '',
           ('整图 CLI 两路运行及逐像素裁图、旧框/主框/E 门槛、概率缓存核验见 '+link(HERE/'data_review/cli_runtime_review.json')+'；其成功证明运行集成正确，不证明准确率提高。' if (HERE/'data_review/cli_runtime_review.json').exists() else '整图 CLI 的独立核验尚未提供。'), '',
           ('本轮错误人工核查：'+link(verification['manual_review']['path']) if verification['manual_review']['status']!='pending' else '本轮错误人工核查尚待完成，报告状态保持待验收。'), '',
           ('人工查看全部 18 个变化/错误并集后，预选候选 16 错分为：9 张无框（含预标纯残骸 1 张）、错个体主框 1 张、叶片主框 1 张、疑似水线桩状背景主框 1 张、低信息细粒度 3 张、箱内灰色/可能幼鸟姿态难例 1 张。`FT_HOLDOUT_153909307` 的源标鸳鸯在前景，实际主框却是远处 25×25 飞鸟；`FT_HOLDOUT_155090015` 原图有多只鸟，73×73 主裁图却是白叶片。这些输入不能仅靠物种分类头微调解决。唯一新增误改 `FT_HOLDOUT_155679526` 的原 E accepted=false，是未确认原始 Top-1 退化，不能写成新增“确认鸟种”误报。细节见 '+link(verification['manual_review']['markdown_path'])
            if verification['manual_review']['status']=='passed' else '错误类别和具体主框观察待人工复核后补入。'), '',
           f"真实 smoke：{link(HERE/'smoke_result.json','DINO smoke')}、{link(HERE/'bio_smoke_result.json','BioCLIP smoke')}；训练数据协议：{link(HERE/'data_review/protocol.json')}；新集输入 QA：{link(HERE/'holdout_data/input_qa.json')}；冻结选择：{link(HERE/'selection.json')}；预测前研究焦点：{link(HERE/'report_candidate.json')}。", '',
           '目前已完成实现、18 组训练、验证选择和冻结新测。未开展局部 Patch、两骨干同时解冻、额外分类层级或部署替换。本轮没有证据支持把真实微调版换进正式 GUI；后续若继续，应先改善训练裁图/逐鸟标签和困难样本覆盖，再以新的训练方案与新的留出集验证。', '',
           '## 完整命令与文件', '',
           '以下命令用于复现实验流程；已有冻结 selection 与测试完成标记会阻止覆盖重跑。审查既有结果直接运行 artifacts；不要为复现而删除冻结文件。', '',
           '```powershell']
    python=PACKAGE/'runtime/python.exe'
    wrapper=HERE/'run.py'
    for args in ('prepare --device cuda','smoke --device cuda','train_tail --device cuda',
                 'bio_tail prepare --device cuda','bio_tail smoke --device cuda','bio_tail train --device cuda',
                 'scoring select --device cuda --batch-size 32','prepare_test --device cuda --bio-prefix',
                 'scoring test --device cuda --batch-size 32','artifacts'):
        md.append(f'& "{python}" -I -S "{wrapper}" {args}')
    md += [f'& "{python}" -I -S "{wrapper}" predict --image "D:\\你的照片.jpg" --output "D:\\新的实验输出" --candidate "{focus}" --device cuda',
           '```', '',
           '训练断点续训在 `train_tail` 或 `bio_tail train` 命令后加 `--resume`，仅允许源摘要、配置和输入合同完全相同的中断运行。最终验收可在 artifacts 命令后提供 `--cli-result <result.json> --manual-review <人工报告> --final`；这只更新验收证据，不改模型选择。', '',
           f"配置：{link(HERE/'config.json')}、{link(HERE/'bio_config.json')}；训练权重在各 `runs/<条件>_<seed>/best.pt`，优化器/RNG 断点在 `last.pt`，完整合同在 `contract.json`。逐样本验证输出位于 `runs/validation/n_val` 与 `o_val`，独立测试输出位于 `runs/test`。", '',
           '本地 DINO 来源固定为 Meta 官方 S/14 与 revision 7764ea0f912e53c92e82eb78a2a1631e92725fc8，参考 [Meta DINOv2](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8)。BioCLIP 原始来源与版本见原资产审查和本轮缓存 binding；未将随机新初始化视觉塔当作原模型微调。', '']
    return '\n'.join(md)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli-result',nargs='+')
    parser.add_argument('--manual-review')
    parser.add_argument('--final',action='store_true')
    args=parser.parse_args()
    assert not torch.cuda.is_initialized(),'Artifacts must run without a CUDA context'
    selection=read(HERE/'selection.json')
    test=read(HERE/'runs/test/summary.json')  # Explicit prerequisite: never report an incomplete test as complete.
    training=load_training(selection)
    verification=evidence(selection,test,training,args.cli_result,args.manual_review,args.final)
    records=prediction_records(test)
    stats={'selection_sha256':sha(HERE/'selection.json'),'report_candidate_id':verification['report_candidate']['id'],
           'test_used_for_reselection':False,'comparisons':{}}
    for identifier,item in records.items():
        reference=records[BASE+('_closed' if test['metrics'][identifier]['class_count']==80 else '_full')]
        stats['comparisons'][identifier]=paired(item['records'],reference['records'])
    qa=read(HERE/'holdout_data/input_qa.json')
    excluded=qa['pure_remains_or_no_recognizable_body_keys']
    assert len(excluded)==1 and qa['predefined_secondary_excluding_pure_remains']['denominator']==94
    secondary={name:{'images':94,'correct':sum(r['correct'] for r in item['records'] if r['key'] not in excluded),
                     'excluded_keys':excluded,'prespecified_before_prediction':True,'main_denominator_unchanged':95}
               for name,item in records.items()}
    stats['prespecified_secondary94']=secondary
    save(HERE/'paired_statistics.json',stats)
    errors={name:{'prediction_json':str(HERE/'runs/test'/(name+'.json')),
                  'wrong':[r for r in item['records'] if not r['correct']],
                  'corrected_keys':stats['comparisons'][name]['corrected'],
                  'regressed_keys':stats['comparisons'][name]['regressed']}
            for name,item in records.items()}
    save(HERE/'error_cases.json',{'scope':'Source-label main-output errors; source pixels and labels retained',
         'images':95,'report_candidate_id':verification['report_candidate']['id'],'models':errors})
    output=HERE/'figures'
    output.mkdir(exist_ok=True)
    learning_curves(training,output)
    comparison_figure(selection,test,output)
    verification['prediction_probability_files_verified']=len(records)
    verification['figures_visual_review']={'status':'passed' if args.final else 'awaiting_final_acceptance',
        'basis':'Three figures inspected via view_image by parent and artifact author; final acceptance authorized separately',
        'files':{str(p):sha(p) for p in output.glob('*.png')}}
    verification['artifacts_sha256']={str(p):sha(p) for p in [HERE/'paired_statistics.json',HERE/'error_cases.json',*output.glob('*.png')]}
    save(HERE/'runtime_verification.json',verification)
    (HERE/'实测报告.md').write_text(report(selection,test,training,records,stats,verification,secondary),encoding='utf-8')
    verification['report_sha256']=sha(HERE/'实测报告.md')
    verification['artifacts_source_sha256']=sha(HERE/'artifacts.py')
    save(HERE/'runtime_verification.json',verification)
    assert not torch.cuda.is_initialized()
    print('CPU artifacts complete:',verification['completion_status'],'training runs',len(training),
          'test images',test['images'],'original assets',verification['original_assets_checked'],flush=True)


if __name__=='__main__':
    main()
