"""Acquire the Meta implementation and small official backbone into this experiment."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'assets'
ASSETS.mkdir(exist_ok=True)
record = ASSETS / 'dinov2.json'
if record.exists():
    raise SystemExit('DINOv2 is already installed in this experiment.')


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'bird-species-research'}), timeout=45)


with request('https://api.github.com/repos/facebookresearch/dinov2/commits/main') as response:
    revision = json.load(response)['sha']
archive_url = f'https://codeload.github.com/facebookresearch/dinov2/zip/{revision}'
with request(archive_url) as response:
    archive = response.read(30 * 1024**2 + 1)
assert len(archive) < 30 * 1024**2
source = ASSETS / 'dinov2_source'
with zipfile.ZipFile(io.BytesIO(archive)) as z:
    for item in z.infolist():
        parts = PurePosixPath(item.filename).parts[1:]
        if not parts or item.is_dir() or not (parts[0] == 'dinov2' or parts[0] in ('hubconf.py', 'LICENSE')):
            continue
        output = source.joinpath(*parts)
        assert output.resolve().is_relative_to(source.resolve())
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(z.read(item))

url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'
destination = ASSETS / 'dinov2_vits14_pretrain.pth'
temporary = destination.with_suffix('.part')
digest = hashlib.sha256()
total = 0
with request(url) as response, temporary.open('wb') as stream:
    while chunk := response.read(1024**2):
        total += len(chunk)
        assert total < 100 * 1024**2
        digest.update(chunk)
        stream.write(chunk)
assert 80 * 1024**2 < total < 100 * 1024**2
temporary.replace(destination)
record.write_text(json.dumps({'revision': revision, 'code_url': archive_url,
    'code_archive_sha256': hashlib.sha256(archive).hexdigest(), 'weight_url': url,
    'weight_bytes': total, 'weight_sha256': digest.hexdigest()}, indent=2), encoding='utf-8')
print(record.read_text(encoding='utf-8'), flush=True)
