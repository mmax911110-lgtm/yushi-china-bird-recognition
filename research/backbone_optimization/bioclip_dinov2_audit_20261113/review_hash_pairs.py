"""Render the audit's candidate pairs for visual confirmation."""
from PIL import Image, ImageDraw, ImageOps
from audit_project import HERE, read

rows = read(HERE / 'similarity_risk_audit.json')['results']['BioCLIP25_P1434']['pairs']
for offset in range(0, len(rows), 6):
    subset = rows[offset:offset + 6]
    canvas = Image.new('RGB', (1260, 200 * len(subset)), 'white')
    draw = ImageDraw.Draw(canvas)
    for local, pair in enumerate(subset):
        top = local * 200
        draw.text((10, top + 8), f"{offset + local + 1:02d} | dHash {pair['hamming_distance']}", fill='black')
        for side, x in [('training', 220), ('holdout', 750)]:
            draw.text((x, top + 8), f"{pair[side + '_key']}  {pair[side + '_species']}", fill='black')
            with Image.open(pair[side + '_path']) as source:
                photo = ImageOps.exif_transpose(source).convert('RGB')
                photo.thumbnail((480, 165))
                canvas.paste(photo, (x, top + 25))
    canvas.save(HERE / f'hash_review_{offset // 6 + 1}.jpg', quality=90)
