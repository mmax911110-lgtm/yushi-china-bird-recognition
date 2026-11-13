"""Aggregate available historic identities for a new held-out collection, CPU only."""
from pathlib import Path
import hashlib
import json
import importlib.util

here=Path(__file__).resolve().parent
root=here.parents[1]
spec=importlib.util.spec_from_file_location('isolation_helpers',root/'detection_accuracy_20261113/data_review/audit_data_isolation.py')
helper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
paths={Path(row['path']) for row in helper.load(root/'bioclip_dinov2_audit_20261113/manifest_audit_v2.json')['manifests']}
for path in tuple(paths):
    for name in ('metadata.json','bird_metadata.json'):
        if (path.parent/name).exists():
            paths.add(path.parent/name)
paths.add(root/'bioclip2_china_eval_20261028/china_metadata.json')
paths |= {root/f'dinov2_species_20261113/frozen_data/{name}.json' for name in ('train','test')}
assert len(paths)==32
paths.add(root/'detection_accuracy_20261113/expanded_test_manifest.json')
blocked={field:set() for field in helper.FIELDS}
coverage=[]
for path in sorted(paths):
    records=list(helper.records(helper.load(path)))
    extracted=helper.values(records)
    for key,values in extracted.items():
        blocked[key].update(values)
    coverage.append({'path':str(path),'sha256':helper.digest(path),'records':len(records)})
payload={'source_files':coverage,'values':{key:sorted((f'{value:016x}' if key=='dhash' else value for value in values),key=str) for key,values in blocked.items()},
         'dhash_representation':'64-bit hexadecimal, normalize before comparison',
         'note':'Includes historical train/validation/test, new155 train and old61 test, all already-scored128; does not certify exhaustive near-duplicate or capture-sequence isolation.'}
destination=here/'blocked_identity_values.json'
destination.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'path':str(destination),'source_files':len(paths),'counts':{key:len(values) for key,values in blocked.items()}},ensure_ascii=False))
