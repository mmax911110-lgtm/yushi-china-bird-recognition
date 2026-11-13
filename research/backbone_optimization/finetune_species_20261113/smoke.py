"""Real cached-crop smoke test for the frozen-prefix/trainable-DINO-tail path."""
import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import HERE, OLD, CLASSES, cache, crop, read, save, sha, state_sha, warmstart, atomic_save, load_npz
from network import build, restore
from encoders import DinoEncoder
from train import load_head


def difference(a, b):
    a, b = a.detach().float(), b.detach().float()
    assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    return {'max_abs_error': float((a-b).abs().max()),
            'mean_abs_error': float((a-b).abs().mean()),
            'minimum_cosine': float(F.cosine_similarity(a, b, dim=-1).min())}


def grad_check(module, expected):
    params = list(module.named_parameters())
    values = [parameter.grad for _, parameter in params]
    if not expected:
        assert all(value is None for value in values)
        return {'parameters': sum(p.numel() for _, p in params), 'all_gradients_none': True}
    assert params and all(value is not None and torch.isfinite(value).all() for value in values)
    squared = sum(float(value.detach().float().square().sum()) for value in values)
    assert math.isfinite(squared) and squared > 0
    return {'parameters': sum(p.numel() for _, p in params), 'finite_gradients': True,
            'gradient_norm': math.sqrt(squared), 'tensors_with_gradients': len(values)}


def optimizer_for(model, config):
    groups = [{'params': list(model.head.parameters()), 'lr': config['head_learning_rate']}]
    if model.train_tail:
        learning_rate = next(item['tail_learning_rate'] for item in config['experiments'] if item['train_tail'])
        groups.append({'params': list(model.block.parameters())+list(model.norm.parameters()),
                       'lr': learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(actual) == len(set(actual)) and set(actual) == expected
    return optimizer


def step(model, optimizer, bio, tokens, target, config):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(bio, tokens)
    assert logits.shape == (len(target), len(CLASSES))
    loss = F.cross_entropy(logits, target, label_smoothing=config['label_smoothing'])
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {'head': grad_check(model.head, True),
                 'block': grad_check(model.block, model.train_tail),
                 'norm': grad_check(model.norm, model.train_tail)}
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                         config['max_gradient_norm'])
    assert torch.isfinite(norm)
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    return {'loss': float(loss.detach()), 'gradients': gradients, 'gradient_norm_before_clip': float(norm)}


def nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(nested_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(nested_equal(x, y) for x, y in zip(a, b))
    return a == b


def run(device):
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261113)
    config = read(HERE/'config.json')
    seed = config['seeds'][0]
    folder = HERE/'smoke'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = read(HERE/'tokens/manifest.json')
    assert sha(HERE/'tokens/binding.json') == manifest['binding_sha256']
    assert manifest['no_augmentation'] and manifest['test_images_encoded'] is False
    for split in config['train_splits']+config['validation']:
        contract = read(HERE/'tokens'/(split+'.json'))
        assert sha(HERE/'tokens'/(split+'.npy')) == contract['tokens_sha256']
    data = cache(config['train_splits'][0])
    rows, tokens, bio, supported = data
    present = np.array([i for i, row in enumerate(rows) if row['bbox'] is not None], dtype=np.int64)
    numeric_indexes = present[:16]
    assert len(numeric_indexes) == 16
    # prepare.py stores its first 16 present crops as one BF16 batch; use that exact batch here.
    assert set(numeric_indexes).issubset(set(supported))
    indexes = supported[:2]
    assert len(indexes) == 2
    x = torch.from_numpy(np.array(bio[indexes], copy=True)).to(device)
    t = torch.from_numpy(np.array(tokens[indexes], copy=True)).to(device)
    target = torch.tensor([int(np.searchsorted(CLASSES, rows[i]['target'])) for i in indexes], device=device)
    assert all(int(CLASSES[int(local)]) == rows[int(i)]['target'] for i, local in zip(indexes, target.cpu()))
    source_weights = OLD/'assets/dinov2_vits14_pretrain.pth'
    original_file_hash = sha(source_weights)
    assert original_file_hash == manifest['source_checkpoint_sha256']
    bio_file = OLD/'features'/(config['train_splits'][0]+'_bio.npz')
    bio_cache_before = sha(bio_file)
    warm_path, warm_state = warmstart(seed)
    warm_file_before = sha(warm_path)
    assert warm_state['global_ids'] == CLASSES.tolist()

    encoder = DinoEncoder(device)
    assert len(encoder.model.blocks) == 12 and not encoder.model.chunked_blocks
    assert encoder.model.num_register_tokens == 0
    assert all(not p.requires_grad for p in encoder.model.parameters())
    encoder_before = state_sha(encoder.model)
    assert encoder_before == manifest['unchanged_full_backbone_sha256']
    model = build(seed, True, device).eval()
    assert all(p.requires_grad for p in model.parameters())
    assert not ({p.data_ptr() for p in model.parameters()} & {p.data_ptr() for p in encoder.model.parameters()})
    original_head = load_head(warm_path, device).eval()
    assert state_sha(model.head) == state_sha(original_head)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    images = [crop(rows[i]) for i in numeric_indexes]
    batch = torch.stack([encoder.preprocess(image) for image in images]).to(device)
    x16 = torch.from_numpy(np.array(bio[numeric_indexes], copy=True)).to(device)
    t16 = torch.from_numpy(np.array(tokens[numeric_indexes], copy=True)).to(device)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=device=='cuda'):
        complete = encoder.model.forward_features(batch)['x_norm_clstoken']
        direct_prefix = encoder.model.prepare_tokens_with_masks(batch)
        for block in encoder.model.blocks[:-1]:
            direct_prefix = block(direct_prefix)
        direct_tail = encoder.model.norm(encoder.model.blocks[-1](direct_prefix))[:, 0]
    full = F.normalize(complete.float(), dim=-1)
    direct_split = F.normalize(direct_tail.float(), dim=-1)
    with torch.no_grad():
        cached_visual = model.visual(t16)
        cached_logits = model(x16, t16)
        full_logits = original_head(x16, full)
    exact_split = difference(direct_split, full)
    prefix_error = difference(t16.flatten(1), direct_prefix.flatten(1))
    persisted_split = difference(cached_visual, full)
    full_head = difference(cached_logits, full_logits)
    # Splitting the SAME batch must preserve the official computation; unlike the old batch-8 cache.
    assert exact_split['max_abs_error'] < 1e-6, exact_split
    assert prefix_error['max_abs_error'] < 1e-5, prefix_error
    assert persisted_split['max_abs_error'] < 1e-6, persisted_split
    assert full_head['max_abs_error'] < 1e-5, full_head
    old_dino = load_npz(OLD/'features'/(config['train_splits'][0]+'_dino.npz'))
    assert old_dino['rows_sha'].item() == sha(OLD/'features'/(config['train_splits'][0]+'.json'))
    old_features = torch.from_numpy(np.array(old_dino['features'][numeric_indexes], copy=True)).to(device)
    with torch.no_grad():
        old_logits = original_head(x16, old_features)
        warm_head_check = original_head(x16, cached_visual)
    old_feature_difference = difference(cached_visual, old_features)
    old_head_difference = difference(cached_logits, old_logits)
    assert old_feature_difference['max_abs_error'] < .015
    # The two warm-start heads must agree on identical embeddings, irrespective of BF16 batch effects.
    assert torch.equal(warm_head_check, cached_logits)
    old_top1_changes = int((old_logits.argmax(1) != cached_logits.argmax(1)).sum())
    del original_head, batch, images, direct_prefix, direct_tail, complete, full, direct_split

    before = {name: state_sha(getattr(model, name)) for name in ('block', 'norm', 'head')}
    optimizer = optimizer_for(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)
    training = step(model, optimizer, x, t, target, config)
    scheduler.step()
    after = {name: state_sha(getattr(model, name)) for name in before}
    assert all(before[name] != after[name] for name in before)
    checkpoint = folder/'tail_after_one_step.pt'
    atomic_save(checkpoint, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'seed': seed, 'train_tail': True,
        'global_ids': CLASSES.tolist(), 'cpu_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state() if device=='cuda' else None})
    model.eval()
    with torch.no_grad():
        saved_logits = model(x, t).clone()
    reloaded, reload_state = restore(checkpoint, device)
    with torch.no_grad():
        reload_logits = reloaded(x, t)
    assert torch.equal(saved_logits, reload_logits)

    # Compare one more real optimizer update after a disk restore to the uninterrupted update.
    # Building restore() consumes RNG; reset the saved states immediately before each branch.
    torch.set_rng_state(reload_state['cpu_rng'])
    if device=='cuda':
        torch.cuda.set_rng_state(reload_state['cuda_rng'])
    second_uninterrupted = step(model, optimizer, x, t, target, config)
    scheduler.step()
    restored_optimizer = optimizer_for(reloaded, config)
    restored_optimizer.load_state_dict(reload_state['optimizer'])
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, 3)
    restored_scheduler.load_state_dict(reload_state['scheduler'])
    torch.set_rng_state(reload_state['cpu_rng'])
    if device=='cuda':
        torch.cuda.set_rng_state(reload_state['cuda_rng'])
    second_resumed = step(reloaded, restored_optimizer, x, t, target, config)
    restored_scheduler.step()
    assert nested_equal(model.state_dict(), reloaded.state_dict())
    assert nested_equal(optimizer.state_dict(), restored_optimizer.state_dict())
    assert nested_equal(scheduler.state_dict(), restored_scheduler.state_dict())
    assert second_uninterrupted['loss'] == second_resumed['loss']

    control = build(seed, False, device)
    assert all(not p.requires_grad for module in (control.block, control.norm) for p in module.parameters())
    control_before = {name: state_sha(getattr(control, name)) for name in before}
    control_optimizer = optimizer_for(control, config)
    control_step = step(control, control_optimizer, x, t, target, config)
    control_after = {name: state_sha(getattr(control, name)) for name in before}
    assert control_before['block'] == control_after['block']
    assert control_before['norm'] == control_after['norm']
    assert control_before['head'] != control_after['head']
    control_checkpoint = folder/'control_after_one_step.pt'
    atomic_save(control_checkpoint, {'model': control.state_dict(), 'seed': seed,
                                     'train_tail': False, 'global_ids': CLASSES.tolist()})
    assert all(p.grad is None for p in encoder.model.parameters())
    encoder_after = state_sha(encoder.model)
    assert encoder_after == encoder_before
    assert sha(source_weights) == original_file_hash and sha(bio_file) == bio_cache_before
    assert sha(warm_path) == warm_file_before

    return {'passed': True, 'device': device, 'seed': seed, 'gradient_training_crops': 2,
        'numerical_comparison_crops': 16, 'first_crop_keys': [rows[int(i)]['key'] for i in indexes],
        'class_count': len(CLASSES), 'global_ids': CLASSES.tolist(), 'class_indices_verified': True,
        'prefix_token_shape': list(t.shape), 'output_shape': list(saved_logits.shape),
        'official_full_vs_direct_split_same_batch': exact_split,
        'persisted_prefix_vs_fresh_prefix_same_batch': prefix_error,
        'official_full_vs_persisted_prefix_and_tail_same_batch': persisted_split,
        'official_full_vs_persisted_prefix_warmhead_same_batch': full_head,
        'old_batch8_features_vs_current_batch16': old_feature_difference,
        'warmhead_old_batch8_vs_current_batch16_logits': old_head_difference,
        'warmhead_old_batch8_vs_current_batch16_top1_changes': old_top1_changes,
        'identical_embedding_warmhead_logits_bitwise_equal': True,
        'tail_training': training, 'tail_before_sha256': before, 'tail_after_sha256': after,
        'control_training': control_step, 'control_before_sha256': control_before,
        'control_after_sha256': control_after, 'control_tail_has_no_gradients_or_updates': True,
        'checkpoint_roundtrip_logits_bitwise_equal': True,
        'resumed_optimizer_update_bitwise_equal': True, 'resumed_optimizer_state_bitwise_equal': True,
        'resumed_scheduler_state_equal': True, 'resume_second_step_loss': second_resumed['loss'],
        'frozen_original_encoder_has_no_gradients': True,
        'frozen_original_encoder_state_before_sha256': encoder_before,
        'frozen_original_encoder_state_after_sha256': encoder_after,
        'original_dino_checkpoint_unchanged_sha256': original_file_hash,
        'frozen_bioclip_train_cache_unchanged_sha256': bio_cache_before,
        'warmstart_checkpoint_sha256': warm_file_before,
        'trainable_parameters_tail_model': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'trainable_parameters_control_model': sum(p.numel() for p in control.parameters() if p.requires_grad),
        'config_sha256': sha(HERE/'config.json'), 'prefix_manifest_sha256': sha(HERE/'tokens/manifest.json'),
        'source_sha256': {name: sha(HERE/name) for name in ('common.py', 'network.py', 'prepare.py', 'smoke.py', 'train_tail.py')},
        'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else None,
        'test_predictions_read': False, 'weights_saved_only_under': str(folder),
        'seconds': time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    destination = HERE/'smoke_result.json'
    save(destination, {'passed': False, 'status': 'running', 'device': args.device})
    try:
        result = run(args.device)
    except Exception as error:
        save(destination, {'passed': False, 'status': 'failed', 'device': args.device,
                           'error': repr(error), 'traceback': traceback.format_exc()})
        raise
    save(destination, result)
    print(json.dumps({'passed': result['passed'], 'device': result['device'],
                      'parameters': result['trainable_parameters_tail_model'],
                      'old_batch8_max_error': result['old_batch8_features_vs_current_batch16']['max_abs_error'],
                      'split_same_batch_max_error': result['official_full_vs_persisted_prefix_and_tail_same_batch']['max_abs_error'],
                      'seconds': result['seconds']}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
