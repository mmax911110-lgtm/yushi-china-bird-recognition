"""Validation-only selection and one frozen independent species test.

Run with ``run.py scoring select`` after all eighteen DINO/Bio runs finish, then
``run.py scoring test`` only after the independent input QA and test caches exist.
This module never changes the detector, source splits, model weights or thresholds.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
from scipy.special import softmax
import torch

from common import (HERE, OLD, PACKAGE, CLASSES, LABELS, read, save, sha, cache,
                    load_npz, warmstart, mapping_sha, state_sha)
from network import build, restore
from evaluate import evaluate, calibration, apply_residual, raw_base


BASELINE = 'deployed_BioCLIP2_2_5'
BASE_NAMES = ('E0_BioCLIP2', BASELINE)
DINO_FAMILIES = ('warmstart', 'head_control', 'tail_lr1e5', 'tail_lr3e5')
BIO_FAMILIES = ('bio_head_control', 'bio_tail_lr1e6', 'bio_tail_lr3e6')
FAMILIES = DINO_FAMILIES + BIO_FAMILIES
SELECTION = HERE / 'selection.json'
FULL_IDS = np.arange(len(LABELS), dtype=np.int64)


def now():
    return datetime.now(timezone.utc).isoformat()


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def mask(rows, supported=False):
    return np.array([row['target'] in CLASSES if supported else row['bbox'] is not None
                     for row in rows], dtype=bool)


def closed_base(probabilities):
    selected = probabilities[:, CLASSES]
    return selected / np.maximum(selected.sum(1, keepdims=True), 1e-300)


def score_key(spec):
    metrics = spec['metrics']
    return (sum(value['correct'] for value in metrics.values()),
            sum(value['macro_f1'] for value in metrics.values()),
            -float(spec.get('alpha', 0.)))


def deployment_passes(candidate, baselines, head_control, stages):
    """Strict global improvement, and no N/O regression against either control."""
    if not candidate.get('selected_tuned_seed_count', 0):
        return False
    measured = candidate['metrics']
    deployed = {stage: baselines[BASELINE][stage]['full'] for stage in stages}
    controls = head_control if isinstance(head_control, list) else [head_control]
    total = sum(measured[stage]['correct'] for stage in stages)
    return (total > sum(deployed[stage]['correct'] for stage in stages)
            and all(total > sum(control['metrics'][stage]['correct'] for stage in stages)
                    for control in controls)
            and all(measured[stage]['correct'] >= deployed[stage]['correct']
                    and all(measured[stage]['correct'] >= control['metrics'][stage]['correct']
                            for control in controls)
                    for stage in stages))


def checked_features(path, rows_path, rows, width):
    values = load_npz(path)
    assert str(values['rows_sha'].item()) == sha(rows_path), path
    features = values['features']
    assert features.shape == (len(rows), width), path
    assert np.isfinite(features).all(), path
    return features


def stage_data(stage):
    """Use common.cache for validation; test rows deliberately live in the new dir."""
    if stage == 'test':
        rows_path = HERE / 'features/test.json'
        rows = read(rows_path)
        contract_path = HERE / 'tokens/test.json'
        contract = read(contract_path)
        metadata = read(HERE / 'tokens/manifest.json')
        assert contract['rows_sha256'] == sha(rows_path)
        assert contract['prefix_binding_sha256'] == metadata['binding_sha256']
        if 'selection_sha256' in contract:
            assert contract['selection_sha256'] == sha(SELECTION)
        assert metadata['source_checkpoint_sha256'] == sha(OLD / 'assets/dinov2_vits14_pretrain.pth')
        token_path = HERE / 'tokens/test.npy'
        assert contract['tokens_sha256'] == sha(token_path)
        tokens = np.load(token_path, mmap_mode='r', allow_pickle=False)
        bio = checked_features(HERE / 'features/test_bio.npz', rows_path, rows, 768)
        bio25_path = HERE / 'features/test_bio25.npz'
    else:
        rows_path = OLD / 'features' / (stage + '.json')
        rows, tokens, bio, _ = cache(stage)
        contract_path = HERE / 'tokens' / (stage + '.json')
        contract = read(contract_path)
        token_path = HERE / 'tokens' / (stage + '.npy')
        assert contract['tokens_sha256'] == sha(token_path)
        bio25_path = OLD / 'features' / (stage + '_bio25.npz')
    assert tokens.shape == (len(rows), 257, 384)
    assert len({row['key'] for row in rows}) == len(rows)
    assert all(0 <= int(row['target']) < len(LABELS) for row in rows)
    present = mask(rows)
    if present.any():
        assert np.isfinite(tokens[present]).all()
    bio25 = checked_features(bio25_path, rows_path, rows, 1024)
    raw = raw_base(bio[present], bio25[present]) if present.any() else {
        name: np.empty((0, len(LABELS)), dtype=np.float64) for name in BASE_NAMES}
    bases = {name: np.full((len(rows), len(LABELS)), 1. / len(LABELS), dtype=np.float64)
             for name in BASE_NAMES}
    for name, values in raw.items():
        bases[name][present] = values
    if stage == 'test':
        bio_contract_path = HERE / 'bio_tokens/test.json'
        bio_contract = read(bio_contract_path)
        bio_metadata = read(HERE / 'bio_tokens/manifest.json')
        assert bio_contract['rows_sha256'] == sha(rows_path)
        assert bio_contract['prefix_binding_sha256'] == bio_metadata['binding_sha256']
        if 'selection_sha256' in bio_contract:
            assert bio_contract['selection_sha256'] == sha(SELECTION)
        bio_token_path = HERE / 'bio_tokens/test.npy'
        assert bio_contract['tokens_sha256'] == sha(bio_token_path)
        bio_tokens = np.load(bio_token_path, mmap_mode='r', allow_pickle=False)
        dino_path = HERE / 'features/test_dino.npz'
        dino = checked_features(dino_path, rows_path, rows, 384)
    else:
        from bio_tail import cache as bio_cache
        bio_rows, bio_tokens, dino, _ = bio_cache(stage)
        assert [row['key'] for row in bio_rows] == [row['key'] for row in rows]
        bio_contract_path = HERE / 'bio_tokens' / (stage + '.json')
        bio_contract = read(bio_contract_path)
        bio_token_path = HERE / 'bio_tokens' / (stage + '.npy')
        assert bio_contract['tokens_sha256'] == sha(bio_token_path)
        dino_path = OLD / 'features' / (stage + '_dino.npz')
    assert bio_tokens.shape == (len(rows), 257, 1024)
    assert dino.shape == (len(rows), 384) and np.isfinite(dino).all()
    if present.any():
        assert np.isfinite(bio_tokens[present]).all()
    return {'rows': rows, 'tokens': tokens, 'bio': bio, 'bases': bases,
            'present': present, 'closed': mask(rows, True), 'rows_path': rows_path,
            'token_contract_path': contract_path, 'token_path': token_path,
            'bio25_path': bio25_path, 'bio_tokens': bio_tokens, 'dino': dino,
            'bio_token_contract_path': bio_contract_path, 'bio_token_path': bio_token_path,
            'dino_path': dino_path}


def checkpoint_evidence(family, seed):
    if family in BIO_FAMILIES:
        return bio_checkpoint_evidence(family, seed)
    if family == 'warmstart':
        path, state = warmstart(seed)
        return {'kind': 'warmstart', 'family': family, 'seed': seed,
                'architecture': 'dinov2_tail',
                'checkpoint': str(path), 'checkpoint_sha256': sha(path),
                'selected_epoch': int(state['epoch']), 'train_tail': False,
                'selected_visual_backbone_updated': False,
                'selected_head_updated': False}
    folder = HERE / 'runs' / f'{family}_{seed}'
    finished = read(folder / 'complete.json')
    contract_path = folder / 'contract.json'
    contract = read(contract_path)
    assert contract['config_sha256'] == sha(HERE / 'config.json')
    assert contract['data_protocol_sha256'] == sha(HERE / 'data_review/protocol.json')
    assert contract['prefix_manifest_sha256'] == sha(HERE / 'tokens/manifest.json')
    assert contract['seed'] == seed and contract['spec']['id'] == family
    assert sha(contract['warmstart_path']) == contract['warmstart_sha256']
    for name, expected in contract['source_sha256'].items():
        assert sha(HERE / name) == expected
    for split, expected in contract['prefix_metadata_sha256'].items():
        assert sha(HERE / 'tokens' / (split + '.json')) == expected
    path = Path(finished['checkpoint'])
    assert sha(path) == finished['checkpoint_sha256']
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert state['global_ids'] == CLASSES.tolist()
    assert state['contract_sha256'] == sha(contract_path)
    assert state['seed'] == seed
    block = mapping_sha({name[len('block.'):]: value for name, value in state['model'].items()
                         if name.startswith('block.')})
    norm = mapping_sha({name[len('norm.'):]: value for name, value in state['model'].items()
                        if name.startswith('norm.')})
    assert block == finished['selected_tail_sha256']
    assert norm == finished['selected_norm_sha256']
    assert state['epoch'] == finished['selected_epoch']
    changed = block != finished['tail_before_sha256'] or norm != finished['norm_before_sha256']
    assert changed == finished['selected_visual_backbone_updated']
    if family == 'head_control':
        assert not state['train_tail'] and not changed
    else:
        assert state['train_tail']
    return {'kind': 'tail_checkpoint', 'family': family, 'seed': seed,
            'architecture': 'dinov2_tail',
            'checkpoint': str(path), 'checkpoint_sha256': sha(path),
            'complete_sha256': sha(folder / 'complete.json'),
            'contract_sha256': sha(contract_path), 'selected_epoch': int(state['epoch']),
            'selected_model_state_sha256': mapping_sha(state['model']),
            'train_tail': bool(state['train_tail']),
            'selected_tail_sha256': block, 'selected_norm_sha256': norm,
            'tail_before_sha256': finished['tail_before_sha256'],
            'norm_before_sha256': finished['norm_before_sha256'],
            'selected_visual_backbone_updated': changed,
            'selected_head_updated': bool(finished['selected_head_updated']),
            'final_epoch_visual_changed': finished['tail_before_sha256'] != finished['tail_after_sha256']
                or finished['norm_before_sha256'] != finished['norm_after_sha256']}


def bio_checkpoint_evidence(family, seed):
    folder = HERE / 'runs' / f'{family}_{seed}'
    finished = read(folder / 'complete.json')
    contract_path = folder / 'contract.json'
    contract = read(contract_path)
    assert contract['architecture'] == 'bioclip_tail'
    assert contract['config_sha256'] == sha(HERE / 'bio_config.json')
    assert contract['data_protocol_sha256'] == sha(HERE / 'data_review/protocol.json')
    assert contract['prefix_manifest_sha256'] == sha(HERE / 'bio_tokens/manifest.json')
    assert contract['seed'] == seed and contract['spec']['id'] == family
    assert sha(contract['warmstart_path']) == contract['warmstart_sha256']
    for name, expected in contract['source_sha256'].items():
        assert sha(HERE / name) == expected
    for split, expected in contract['prefix_metadata_sha256'].items():
        assert sha(HERE / 'bio_tokens' / (split + '.json')) == expected
    path = Path(finished['checkpoint'])
    assert sha(path) == finished['checkpoint_sha256']
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert state['architecture'] == 'bioclip_tail' and state['global_ids'] == CLASSES.tolist()
    assert state['contract_sha256'] == sha(contract_path) and state['seed'] == seed
    assert state['source_checkpoint_sha256'] == contract['source_checkpoint_sha256']
    selected = {
        'tail': mapping_sha({name[len('block.'):]: value for name, value in state['model'].items()
                             if name.startswith('block.')}),
        'norm': mapping_sha({name[len('ln_post.'):]: value for name, value in state['model'].items()
                             if name.startswith('ln_post.')}),
        'proj': mapping_sha({'proj': state['model']['proj']})}
    changed_parts = {}
    for part, digest in selected.items():
        assert digest == finished[f'selected_{part}_sha256']
        changed_parts[part] = digest != finished[f'{part}_before_sha256']
        assert changed_parts[part] == finished[f'selected_{part}_updated']
    changed = any(changed_parts.values())
    assert changed == finished['selected_visual_backbone_updated']
    assert state['epoch'] == finished['selected_epoch']
    if family == 'bio_head_control':
        assert not state['train_tail'] and not changed
    else:
        assert state['train_tail']
    return {'kind': 'tail_checkpoint', 'architecture': 'bioclip_tail',
            'family': family, 'seed': seed, 'checkpoint': str(path),
            'checkpoint_sha256': sha(path), 'complete_sha256': sha(folder / 'complete.json'),
            'contract_sha256': sha(contract_path), 'selected_epoch': int(state['epoch']),
            'selected_model_state_sha256': mapping_sha(state['model']),
            'train_tail': bool(state['train_tail']),
            **{f'selected_{part}_sha256': digest for part, digest in selected.items()},
            **{f'{part}_before_sha256': finished[f'{part}_before_sha256'] for part in selected},
            **{f'selected_{part}_updated': changed_parts[part] for part in selected},
            'selected_visual_backbone_updated': changed,
            'selected_head_updated': bool(finished['selected_head_updated']),
            'final_epoch_visual_changed': any(finished[f'{part}_before_sha256'] != finished[f'{part}_after_sha256']
                                             for part in selected)}


def family_logits(models, stages, device, batch_size):
    """Infer every detected validation image, including unsupported species."""
    outputs = {stage: [] for stage in stages}
    timing = []
    for item in models:
        assert sha(item['checkpoint']) == item['checkpoint_sha256']
        is_bio = item.get('architecture') == 'bioclip_tail'
        if item['kind'] == 'warmstart':
            model = build(item['seed'], False, device).eval()
        else:
            if is_bio:
                from bio_tail import restore as bio_restore
                model, state = bio_restore(item['checkpoint'], device)
            else:
                model, state = restore(item['checkpoint'], device)
            assert state['epoch'] == item['selected_epoch']
            assert state_sha(model) == item['selected_model_state_sha256']
        model.requires_grad_(False)
        for name, stage in stages.items():
            logits = np.zeros((len(stage['rows']), len(CLASSES)), dtype=np.float64)
            indexes = np.flatnonzero(stage['present'])
            if device == 'cuda':
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                for begin in range(0, len(indexes), batch_size):
                    selected = indexes[begin:begin + batch_size]
                    if is_bio:
                        tokens = torch.from_numpy(np.array(stage['bio_tokens'][selected], copy=True)).to(device)
                        x = torch.from_numpy(np.array(stage['dino'][selected], copy=True)).to(device)
                        predicted = model(tokens, x)
                    else:
                        x = torch.from_numpy(np.array(stage['bio'][selected], copy=True)).to(device)
                        tokens = torch.from_numpy(np.array(stage['tokens'][selected], copy=True)).to(device)
                        predicted = model(x, tokens)
                    logits[selected] = predicted.detach().float().cpu().numpy().astype(np.float64)
            if device == 'cuda':
                torch.cuda.synchronize()
            assert np.isfinite(logits).all()
            outputs[name].append(logits)
            timing.append({'seed': item['seed'], 'stage': name, 'images': len(indexes),
                           'elapsed_seconds': time.perf_counter() - started,
                           'architecture': item.get('architecture'),
                           'scope': 'cached final visual block/norm/projection and species head only; excludes detector, crop and frozen encoder prefixes'})
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    return outputs, timing


def baseline_metrics(stages):
    return {name: {stage_name: {
        'full': evaluate(stage['rows'], stage['bases'][name], FULL_IDS),
        'closed': evaluate([row for i, row in enumerate(stage['rows']) if stage['closed'][i]],
                           closed_base(stage['bases'][name][stage['closed']]), CLASSES)}
        for stage_name, stage in stages.items()} for name in BASE_NAMES}


def measurements(spec, averaged, stages):
    output = {}
    for name, stage in stages.items():
        if spec['form'] == 'closed':
            choose = stage['closed']
            rows = [row for i, row in enumerate(stage['rows']) if choose[i]]
            probabilities = softmax(averaged[name][choose] / spec['temperature'], axis=1)
            classes = CLASSES
        else:
            rows = stage['rows']
            probabilities = apply_residual(stage['bases'][BASELINE], averaged[name],
                                           spec['alpha'], spec['temperature'])
            classes = FULL_IDS
        output[name] = evaluate(rows, probabilities, classes)
    return output


def current_bindings(config):
    paths = [HERE / name for name in ('config.json', 'common.py', 'network.py',
                                     'train_tail.py', 'scoring.py', 'data_review/protocol.json',
                                     'tokens/manifest.json', 'training_complete.json')]
    paths += [HERE / name for name in ('bio_config.json', 'bio_tail.py', 'bio_tokens/manifest.json',
                                      'bio_training_complete.json')]
    for split in config['train_splits'] + config['validation']:
        paths += [OLD / 'features' / (split + '.json'),
                  OLD / 'features' / (split + '_bio.npz'),
                  OLD / 'features' / (split + '_dino.npz'),
                  HERE / 'tokens' / (split + '.json'),
                  HERE / 'bio_tokens' / (split + '.json')]
    for split in config['validation']:
        paths.append(OLD / 'features' / (split + '_bio25.npz'))
    return {str(path): sha(path) for path in paths}


def assert_selection_inputs(selection):
    for path, expected in selection['frozen_input_sha256'].items():
        assert sha(path) == expected, f'Frozen selection input changed: {path}'


def checked_training(config):
    expected = {f"{spec['id']}_{seed}" for spec in config['experiments'] for seed in config['seeds']}
    assert len(expected) == 9 and set(spec['id'] for spec in config['experiments']) == set(DINO_FAMILIES[1:])
    if not (HERE / 'bio_config.json').exists():
        raise RuntimeError('Unified selection waits for the independent BioCLIP fine-tuning stage.')
    bio_config = read(HERE / 'bio_config.json')
    assert bio_config['seeds'] == config['seeds']
    assert set(spec['id'] for spec in bio_config['experiments']) == set(BIO_FAMILIES)
    expected |= {f"{spec['id']}_{seed}" for spec in bio_config['experiments'] for seed in bio_config['seeds']}
    assert len(expected) == 18
    missing = [name for name in sorted(expected) if not (HERE / 'runs' / name / 'complete.json').exists()]
    if missing or not (HERE / 'training_complete.json').exists() or not (HERE / 'bio_training_complete.json').exists():
        raise RuntimeError('Unified selection waits for all eighteen completed DINO/Bio runs; missing: ' + ', '.join(missing))
    assert read(HERE / 'data_review/protocol.json')['training_can_start']


def write_prediction(path, rows, probabilities, classes, reference, extra=None, other_references=None):
    """Sparse JSON diagnostics and all-class per-image probabilities in bound NPZ."""
    result = evaluate(rows, probabilities, classes, True)
    detail = result.pop('rows')
    assert len(detail) == len(reference) and all(a['key'] == b['key'] for a, b in zip(detail, reference))
    corrected = [a['key'] for a, b in zip(detail, reference) if a['correct'] and not b['correct']]
    regressed = [a['key'] for a, b in zip(detail, reference) if not a['correct'] and b['correct']]
    paired = {}
    for name, other in (other_references or {}).items():
        assert len(detail) == len(other) and all(a['key'] == b['key'] for a, b in zip(detail, other))
        paired[name] = {
            'corrected': [a['key'] for a, b in zip(detail, other) if a['correct'] and not b['correct']],
            'regressed': [a['key'] for a, b in zip(detail, other) if not a['correct'] and b['correct']]}
    present = np.array([row['has_box'] for row in detail], dtype=bool)
    probability_path = path.with_suffix('.npz')
    save_npz(probability_path, probabilities=np.where(present[:, None], probabilities, 0.),
             class_ids=classes, present=present, keys=np.array([row['key'] for row in rows]),
             targets=np.array([row['target'] for row in rows], dtype=np.int64),
             predictions=np.array([row['prediction'] for row in detail], dtype=np.int64))
    save(path, {'records': detail, **result, 'corrected': corrected, 'regressed': regressed,
                'paired_vs_controls': paired,
                'probabilities': {'path': str(probability_path), 'sha256': sha(probability_path),
                                  'rows_follow_records': True, 'missing_crop_probabilities_are_zero': True},
                **(extra or {})})
    compact = {key: value for key, value in result.items() if key not in ('confusion', 'per_species')}
    compact.update(corrected=len(corrected), regressed=len(regressed), class_count=len(classes))
    compact['paired_vs_controls'] = {name: {kind: len(keys) for kind, keys in pair.items()}
                                    for name, pair in paired.items()}
    return compact, detail


def write_validation_outputs(candidates, all_averaged, stages):
    summary = {}
    for name, stage in stages.items():
        closed_rows = [row for i, row in enumerate(stage['rows']) if stage['closed'][i]]
        reference_full = evaluate(stage['rows'], stage['bases'][BASELINE], FULL_IDS, True)['rows']
        reference_closed = evaluate(closed_rows, closed_base(stage['bases'][BASELINE][stage['closed']]), CLASSES, True)['rows']
        comparisons = {}
        for baseline_name, values in stage['bases'].items():
            comparisons[baseline_name + '_full'] = (stage['rows'], values, FULL_IDS, reference_full, None)
            comparisons[baseline_name + '_closed'] = (closed_rows, closed_base(values[stage['closed']]), CLASSES, reference_closed, None)
        for spec in candidates:
            logits = all_averaged[spec['family']][name]
            if spec['form'] == 'closed':
                measured_rows, classes, reference = closed_rows, CLASSES, reference_closed
                probabilities = softmax(logits[stage['closed']] / spec['temperature'], axis=1)
            else:
                measured_rows, classes, reference = stage['rows'], FULL_IDS, reference_full
                probabilities = apply_residual(stage['bases'][BASELINE], logits, spec['alpha'], spec['temperature'])
            comparisons[spec['id']] = (measured_rows, probabilities, classes, reference, spec)
        summary[name] = {}
        for identifier, (measured_rows, probabilities, classes, reference, spec) in comparisons.items():
            compact, _ = write_prediction(HERE / 'runs/validation' / name / (identifier + '.json'),
                measured_rows, probabilities, classes, reference,
                {'spec': spec, 'used_for_validation_selection': True,
                 'source_rows_sha256': sha(stage['rows_path'])})
            summary[name][identifier] = compact
    save(HERE / 'runs/validation/summary.json', {'metrics': summary, 'test_used': False})


def select(device='cuda', batch_size=32):
    if SELECTION.exists():
        raise RuntimeError('Selection is already frozen; preserve it.')
    config = read(HERE / 'config.json')
    checked_training(config)
    assert config['validation'] == ['n_val', 'o_val']
    stages = {name: stage_data(name) for name in config['validation']}
    assert sum(len(stage['rows']) for stage in stages.values()) == 388
    assert sum(stage['present'].sum() for stage in stages.values()) == 374
    assert sum(stage['closed'].sum() for stage in stages.values()) == 312
    assert sum((stage['closed'] & stage['present']).sum() for stage in stages.values()) == 302
    baselines = baseline_metrics(stages)
    families, closed_trials, residual_trials, candidates, timings = {}, [], [], [], {}
    all_averaged = {}
    for family in FAMILIES:
        models = [checkpoint_evidence(family, seed) for seed in config['seeds']]
        outputs, timing = family_logits(models, stages, device, batch_size)
        averaged = {name: np.mean(values, axis=0) for name, values in outputs.items()}
        all_averaged[family] = averaged
        fit_logits, fit_targets = [], []
        for name, stage in stages.items():
            valid = stage['closed'] & stage['present']
            fit_logits.append(averaged[name][valid])
            fit_targets.extend(np.searchsorted(CLASSES, [row['target'] for i, row in enumerate(stage['rows']) if valid[i]]))
        assert len(fit_targets) == 302
        temperature = calibration(np.concatenate(fit_logits), np.asarray(fit_targets, dtype=np.int64))
        base_spec = {'family': family, 'models': models, 'seed_tag': 'all',
                     'architecture': models[0]['architecture'],
                     'temperature': float(temperature), 'temperature_fit_images': 302,
                     'selected_tuned_seed_count': sum(model['selected_visual_backbone_updated'] for model in models)}
        closed = {**base_spec, 'id': family + '_closed', 'form': 'closed'}
        closed['metrics'] = measurements(closed, averaged, stages)
        closed_trials.append(closed)
        this_family = []
        family_config = read(HERE / 'bio_config.json') if family in BIO_FAMILIES else config
        for alpha in family_config['residual_weights']:
            trial = {**base_spec, 'id': family + f'_full_residual{alpha:g}',
                     'form': 'full_residual', 'alpha': float(alpha)}
            trial['metrics'] = measurements(trial, averaged, stages)
            this_family.append(trial)
            residual_trials.append(trial)
        best_full = max(this_family, key=score_key)
        candidates.extend([closed, best_full])
        families[family] = {'models': models, 'temperature': temperature,
                            'architecture': models[0]['architecture'],
                            'closed_comparison': closed['id'], 'full_comparison': best_full['id'],
                            'selected_tuned_seed_count': base_spec['selected_tuned_seed_count']}
        timings[family] = timing
        print('VALIDATION', family, 'closed', score_key(closed)[0], '/312',
              'full', score_key(best_full)[0], '/388', 'alpha', best_full['alpha'],
              'selected tuned seeds', base_spec['selected_tuned_seed_count'], flush=True)
    head_controls = [spec for spec in candidates if spec['family'] in ('head_control', 'bio_head_control')
                     and spec['form'] == 'full_residual']
    passing = [spec for spec in residual_trials if (spec['family'].startswith('tail_') or spec['family'].startswith('bio_tail_'))
               and spec['form'] == 'full_residual'
               and deployment_passes(spec, baselines, head_controls, stages)]
    deployment = max(passing, key=score_key) if passing else None
    if deployment is not None and deployment['id'] not in {spec['id'] for spec in candidates}:
        candidates.append(deployment)
    # Freeze all seven fair research controls even when deployment guard fails.
    selection = {'status': 'validation_selection_frozen', 'utc': now(),
                 'candidates': candidates, 'families': families, 'deployment_candidate': deployment,
                 'deployment_candidate_id': None if deployment is None else deployment['id'],
                 'development_baselines': baselines, 'all_closed_trials': closed_trials,
                 'all_residual_trials': residual_trials, 'validation_timings': timings,
                 'validation_counts': {'full': 388, 'detected_full': 374, 'supported': 312,
                                       'supported_detected_temperature_fit': 302},
                 'deployment_guard': 'Selected actual tail update; full388 strictly exceeds deployed and both DINO/Bio head_controls; N/O individually no lower than all three controls.',
                 'research_scope': 'Seven three-seed ensembles across separately tuned DINO/Bio backbones, fixed before fresh test; no test-driven reselection.',
                 'frozen_input_sha256': current_bindings(config),
                 'test_features_read': False, 'old61_or128_used_for_selection': False,
                 'new_holdout_used_for_selection': False,
                 'test_input_paths': ['holdout_data/final_manifest.json', 'holdout_data/final_test.json',
                                      'holdout_data/input_qa.json', 'features/test.json']}
    write_validation_outputs(candidates, all_averaged, stages)
    save(SELECTION, selection)
    print('SELECTION_FROZEN', 'deployment', selection['deployment_candidate_id'], flush=True)


def checked_holdout(selection):
    folder = HERE / 'holdout_data'
    manifest_path, source_path, qa_path = (folder / name for name in ('final_manifest.json', 'final_test.json', 'input_qa.json'))
    manifest, source, qa = read(manifest_path), read(source_path), read(qa_path)
    assert manifest['test_sha256'] == sha(source_path)
    assert manifest.get('model_outputs_used', manifest.get('any_model_inference_performed')) is False
    assert manifest['training_or_hyperparameter_selection_forbidden'] is True
    assert qa['manifest_sha256'] == sha(manifest_path) and qa['test_sha256'] == sha(source_path)
    assert qa['new_inference_can_start'] is True and qa['reviewed_before_inference'] is True
    assert qa['training_or_selection_used'] is False
    assert qa['source_labels_changed'] is False and qa['denominator_changed'] is False
    assert len(source) == qa['images'] == manifest['images'] and len(source) > 0
    rows_path = HERE / 'features/test.json'
    rows = read(rows_path)
    assert len(rows) == len(source)
    for original, row in zip(source, rows):
        assert original['split'] == 'test'
        assert original['key'] == row['key'] and original['sha256'] == row['sha256']
        assert str(original['user_id']) == str(row['user_id'])
        assert LABELS[int(row['target'])]['scientific_name'] == original['scientific_name']
        assert row['target'] in CLASSES
        assert sha(row['path']) == row['sha256']
    assert len({str(row['user_id']) for row in source}) == len(source)
    for item in manifest['source_files']:
        assert sha(item['path']) == item['sha256']
    return {str(path): sha(path) for path in (manifest_path, source_path, qa_path, rows_path)}, source


def test(device='cuda', batch_size=32):
    selection = read(SELECTION)
    assert_selection_inputs(selection)
    destination = HERE / 'runs/test'
    if (destination / 'summary.json').exists() or (destination / 'scoring_started.json').exists():
        raise RuntimeError('Independent test scoring already started/completed. Preserve outputs and diagnose any interrupted run before an explicit recovery.')
    bindings, source = checked_holdout(selection)
    # All gate checks precede any fresh inference. A marker prevents accidental retries.
    destination.mkdir(parents=True, exist_ok=True)
    save(destination / 'scoring_started.json', {'utc': now(), 'selection_sha256': sha(SELECTION),
                                               'test_input_sha256': bindings})
    stage = stage_data('test')
    rows = stage['rows']
    assert all(stage['closed']) and len(rows) == len(source)
    test_cache_paths = [stage['token_contract_path'], stage['token_path'],
                        HERE / 'features/test_bio.npz', HERE / 'features/test_bio25.npz',
                        stage['bio_token_contract_path'], stage['bio_token_path'], stage['dino_path']]
    bindings.update({str(path): sha(path) for path in test_cache_paths})
    output = {}
    for name, probabilities in stage['bases'].items():
        output[name + '_full'] = (probabilities, FULL_IDS, None)
        output[name + '_closed'] = (closed_base(probabilities), CLASSES, None)
    averaged_by_family, individual_by_family, timings = {}, {}, {}
    for family in FAMILIES:
        models = selection['families'][family]['models']
        individual, timing = family_logits(models, {'test': stage}, device, batch_size)
        individual_by_family[family] = individual['test']
        averaged_by_family[family] = np.mean(individual['test'], axis=0)
        timings[family] = timing
    for spec in selection['candidates']:
        logits = averaged_by_family[spec['family']]
        if spec['form'] == 'closed':
            probabilities, classes = softmax(logits / spec['temperature'], axis=1), CLASSES
        else:
            probabilities = apply_residual(stage['bases'][BASELINE], logits, spec['alpha'], spec['temperature'])
            classes = FULL_IDS
        output[spec['id']] = (probabilities, classes, spec)
    references = {len(classes): evaluate(rows, probabilities, classes, True)['rows']
                  for probabilities, classes, _ in (output[BASELINE + '_full'], output[BASELINE + '_closed'])}
    control_specs = [spec for spec in selection['candidates'] if spec['family'] in ('head_control', 'bio_head_control')]
    control_rows = {}
    for spec in control_specs:
        probabilities, classes, _ = output[spec['id']]
        control_rows.setdefault(len(classes), {})[spec['family']] = evaluate(rows, probabilities, classes, True)['rows']
    result, selected_seed_results = {}, {}
    for name, (probabilities, classes, spec) in output.items():
        extra = {'source_rows_sha256': sha(HERE / 'features/test.json'),
                 'selection_sha256': sha(SELECTION), 'spec': spec}
        measured, detail = write_prediction(destination / (name + '.json'), rows, probabilities,
                                            classes, references[len(classes)], extra, control_rows[len(classes)])
        result[name] = measured
        if spec is not None:
            fixed_seed_metrics = []
            for model, logits in zip(spec['models'], individual_by_family[spec['family']]):
                if spec['form'] == 'closed':
                    values = softmax(logits / spec['temperature'], axis=1)
                else:
                    values = apply_residual(stage['bases'][BASELINE], logits, spec['alpha'], spec['temperature'])
                fixed_seed_metrics.append({'seed': model['seed'], 'selected_epoch': model['selected_epoch'],
                                           'selected_visual_backbone_updated': model['selected_visual_backbone_updated'],
                                           'metrics': evaluate(rows, values, classes)})
            selected_seed_results[name] = {'note': 'Individual seeds use the same ensemble-frozen temperature/alpha, not per-seed test tuning.',
                                           'seeds': fixed_seed_metrics,
                                           'mean_top1': float(np.mean([m['metrics']['top1'] for m in fixed_seed_metrics])),
                                           'std_top1': float(np.std([m['metrics']['top1'] for m in fixed_seed_metrics]))}
        print('TEST', name, measured['correct'], '/', len(rows), 'corrected', measured['corrected'],
              'regressed', measured['regressed'], flush=True)
    save(destination / 'per_seed_fixed_parameters.json', selected_seed_results)
    save(destination / 'summary.json', {'status': 'completed_frozen_independent_test', 'utc': now(),
         'metrics': result, 'selection_sha256': sha(SELECTION), 'test_input_sha256': bindings,
         'images': len(rows), 'species': len({row['target'] for row in rows}),
         'deployment_candidate_id': selection['deployment_candidate_id'],
         'deployment_candidate_changed_after_test': False,
         'test_retuning': False, 'fixed_detector_and_gate': True,
         'no_box': int((~stage['present']).sum()), 'timings': timings,
         'per_seed_metrics': str(destination / 'per_seed_fixed_parameters.json'),
         'probability_scope': 'All class probabilities saved per sample in bound NPZ files; no-box rows are zeros.',
         'old61_or128_used_for_selection': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('select', 'test'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    assert args.batch_size > 0
    torch.set_num_threads(4)
    {'select': select, 'test': test}[args.stage](args.device, args.batch_size)


if __name__ == '__main__':
    main()
