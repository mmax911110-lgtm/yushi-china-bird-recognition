"""Check the candidate manifest and verify existing deployment files again."""
from collections import Counter
from pathlib import Path
import json
from audit_project import HERE, WORK, PACKAGE, read, save, sha
from audit_data import extract

manifests = read(HERE / 'manifest_audit_v2.json')['manifests']
known = set()
for info in manifests:
    for row in extract(read(info['path'])):
        digest = row.get('sha256') or row.get('source_sha256')
        if digest:
            known.add(digest)

rows = [json.loads(line) for line in (HERE / 'additional_training_candidates.jsonl').read_text(encoding='utf-8').splitlines()]
assets = read(HERE / 'protected_assets_before.json')
assert all(row['official_split'] == 'train' for row in rows)
assert all(row['path_exists'] and Path(row['path']).is_file() for row in rows)
candidate_digests = [sha(row['path']) for row in rows]
known_duplicates = [row['image_id'] for row, digest in zip(rows, candidate_digests) if digest in known]
internal_counts = Counter(candidate_digests)
save('additional_candidate_sha_audit.json', {
    'candidates_hashed': len(rows), 'known_manifest_hashes': len(known),
    'exact_cross_manifest_duplicate_ids': known_duplicates,
    'internal_sha_duplicate_groups': sum(count > 1 for count in internal_counts.values()),
    'candidate_sha256_by_CUB_image_id': {str(row['image_id']): digest for row, digest in zip(rows, candidate_digests)},
    'mapping_still_pending_review': True, 'near_duplicates_not_screened_for_CUB': True,
})

pair_review = read(HERE / 'similarity_risk_audit.json')['results']['BioCLIP25_P1434']['pairs']
save('manual_similarity_review.json', {
    'pairs_reviewed': len(pair_review), 'visually_confirmed_near_duplicate_pairs': [],
    'sheets': [f'hash_review_{index}.jpg' for index in (1, 2, 3)],
    'scope': 'Visual review of the 17 metadata dHash flags. Similar low-texture sky, water and wall images dominate. No duplicate confirmed in these flags. Does not establish capture-sequence disjointness.',
})

for value in assets.values():
    assert sha(value['path']) == value['sha256'], value['path']
runtime = read(HERE / 'runtime_probe.json')
assert not runtime['cuda_context_initialized']
summary = read(HERE / 'cached_baseline_summary.json')['outputs']
assert summary['p_test']['BioCLIP2_E']['correct'] == 169
assert summary['p_test']['deployed_BioCLIP2_plus25']['correct'] == 172
assert summary['previous158']['deployed_BioCLIP2_plus25']['correct'] == 139
assert summary['previous129']['deployed_BioCLIP2_plus25']['correct'] == 111
save('audit_verification.json', {
    'status': 'review_ready', 'protected_asset_hashes_verified_again': len(assets),
    'cached_baseline_historical_top1_reconciled': True,
    'source_images_downloaded': 0, 'new_model_weights_downloaded': 0,
    'new_model_training_started': False, 'new_gpu_model_inference': False,
    'CUB_training_candidates': len(rows), 'CUB_exact_cross_manifest_duplicates': len(known_duplicates),
    'CUB_internal_exact_duplicate_groups': sum(count > 1 for count in internal_counts.values()),
})
print('AUDIT VERIFIED', len(assets), 'protected assets;', len(rows), 'CUB candidates;',
      len(known_duplicates), 'known exact duplicates', flush=True)
