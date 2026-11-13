"""Finalize CPU-only input review and the permitted fine-tuning data contract."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'dinov2_species_20261113'
BIO=ROOT/'bioclip2_china_eval_20261028'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


audit=read(HERE/'finetune_data_audit.json')
assert not any(any(values.values()) for values in audit['overlaps'].values())
assert audit['training_validation_image_SHA256_verified']
assert audit['groups']['train_expanded1577']['images']==1577
assert audit['groups']['validation_supported312']['cropped_images']==302
near_review={
    'reviewed_utc':datetime.now(timezone.utc).isoformat(),
    'source_audit_sha256':sha(HERE/'finetune_data_audit.json'),
    'method':'Human visual inspection of both original photos with view_image before new fine-tuning.',
    'candidates':[{
        'train_key':'I_260988924','validation_key':'N_258522537','dhash_hamming':4,
        'train_path':str(BIO/'phase_i/data/I_260988924.jpg'),
        'validation_path':str(BIO/'phase_n_accuracy_20261106/fresh_data/data/N_258522537.jpg'),
        'visual_conclusion':'different_photos_similar_waterfowl_subject',
        'observations':'Training photo has a darker exposure, multiple ducks of different poses and visible shore; validation photo shows two swimming ducks, brighter water and different framing. No shared image content indicating a crop/re-encode was observed.',
        'same_photographer':False,'exact_SHA_or_photo_or_observation_overlap':False,
        'action':'keep_both_original_splits','split_changed':False,'labels_changed':False,
        'sequence_independence_certified':False,
    }],
    'near_duplicate_screening_limits':'64-bit dHash can collide for similar water/sky scenes; this is not exhaustive sequence or foundation-training decontamination.'
}
(HERE/'nearhash_manual_review.json').write_text(json.dumps(near_review,ensure_ascii=False,indent=2),encoding='utf-8')
source_paths=[OLD/f'features/{name}.json' for name in ('train','expanded','n_val','o_val')]
source_paths += [OLD/'frozen_data/train.json',BIO/'phase_p_bioclip25_20261106/support_cache/records.json']
validation_sources=[BIO/f'phase_{phase}/fresh_data/manifest.json' for phase in ('n_accuracy_20261106','o_offline_accuracy_20261106')]
proposal_names=['train_proposal.json','train_expanded1577_proposal.json','validation388_proposal.json','validation_supported312_proposal.json','validation_unsupported76_proposal.json','class_index80.json']
protocol={
    'status':'CPU_input_review_passed_training_permitted',
    'frozen_utc':datetime.now(timezone.utc).isoformat(),
    'training_can_start':True,
    'scope':'Paired fixed BioCLIP2 features + real DINOv2 final-block/norm fine-tuning, compared to an identical head-only control. Original YOLO and original model weights remain protected.',
    'audit_sha256':sha(HERE/'finetune_data_audit.json'),
    'nearhash_manual_review_sha256':sha(HERE/'nearhash_manual_review.json'),
    'training_and_validation_source_sha256':{str(path):sha(path) for path in source_paths+validation_sources},
    'proposal_sha256':{name:sha(HERE/name) for name in proposal_names},
    'train':{'feature_splits':['train','expanded'],'original1434':1434,'extra_source155':155,'extra_usable143':143,
             'usable_images':1577,'species':80,'photographers':747,'original_split_roles_retained':True,
             'rule':'bbox is not None and global target belongs to the existing supported80. No filtering by accepted/correctness or validation/test results.',
             'no_box_source_records_excluded_from_this_crop_protocol':12},
    'validation':{'feature_splits':['n_val','o_val'],'full_images':388,'full_species':97,
                  'supported_closed_images':312,'supported_species_present':78,'supported_with_crops':302,
                  'supported_missing_crops':10,'unsupported_images':76,'unsupported_species':19,
                  'full_with_crops':374,'full_missing_crops':14,
                  'supported_species_absent_from_validation':audit['training_species_absent_from_validation'],
                  'selection':'Only existing N/O development validation. 80-way head uses the fixed supported312 denominator; absent detections count wrong for the end-to-end closed metric. Conditional crop-only result uses302 and must be labeled conditional.',
                  'full_vocab_reporting':'Full388 requires frozen global-head/residual/fallback protocol. Do not feed unsupported global targets into an80-way loss or call the closed subset388.',
                  'prior_reuse':'These validation images have been used in historical model selection. They are independent of current training identities, not a pristine external test.'},
    'fair_control':{'same_train_val_crops_global_ids_seeds_warmstart_and_head_budget':True,
                    'only_major_difference':'DINOv2 last block and final norm are trainable in tail candidates; they are frozen in head_control.',
                    'required_reporting':['head_control','tail_lr1e5','tail_lr3e5','three_seed_mean_and_spread','paired corrected/regressed validation images']},
    'heldout_restrictions':{
        'old61':{'path':str(OLD/'frozen_data/test.json'),'sha256':sha(OLD/'frozen_data/test.json'),'images':61,'role':'already-scored locked regression only'},
        'old128':{'path':str(ROOT/'detection_accuracy_20261113/expanded_test_manifest.json'),'sha256':sha(ROOT/'detection_accuracy_20261113/expanded_test_manifest.json'),'images':128,'role':'already-scored locked regression only'},
        'never_use_old61_or_old128_for':['gradient training','checkpoint/epoch selection','learning-rate or augmentation selection','residual/threshold/policy changes','choosing between experimental architectures'],
        'new_holdout':'Collect and freeze separately against blocked_identity_values.json. No scoring until all model/hyperparameter choices freeze. Any later scoring is reporting only; do not tune on those results.',
        'current_known_untouched_final_test':False,
    },
    'identity_checks':{'train_vs_validation_key_user_observation_photo_SHA_exact_dhash_overlap':0,
                       'train_vs61_and128_overlap':0,'within_train_or_validation_duplicate_key_photo_observation_SHA_exact_dhash':0,
                       'all_proposed_training_validation_source_files_SHA_verified':True,
                       'verified_unique_source_images':1965,'verified_bytes':audit['verified_source_bytes'],
                       'nearhash_candidates':1,'manual_review':'different photos, retain both, no split changed'},
    'data_limitations':['80 training species have4 to33 usable images each; class counts remain small and imbalanced.',
                        'Selected detector crops use observation-level species labels rather than per-box expert identity.',
                        'Capture-sequence independence and foundation-pretraining overlap are not exhaustively known.',
                        'No original labels, source split, weights, photos, or old reports were changed by this audit.'],
}
config=HERE.parent/'config.json'
if config.exists():
    protocol['reviewed_training_config_snapshot_sha256']=sha(config)
(HERE/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'training_can_start':protocol['training_can_start'],'train_images':1577,'closed_validation_denominator':312,
                  'closed_validation_crops':302,'full_validation_denominator':388,'nearhash_review':'different_photos_keep_both'},ensure_ascii=False))
