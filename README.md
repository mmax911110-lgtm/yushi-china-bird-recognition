# Yushi China Bird Recognition Model

**YOLO11 · BioCLIP · Offline inference · Computer vision**

English · [简体中文](README.zh-CN.md) · [日本語](README.ja.md)

Yushi (羽识) detects birds in photographs and suggests a species for each retained crop. It focuses on birds found in China and ranks candidates from a reference list of **1,343 species**. The Windows app runs locally and shows the detection box, the actual classifier crop, ranked species suggestions, and whether the result needs review.

![Yushi workflow on a Little Grebe photograph](media/little-grebe-workflow.png)

[Grey Heron example](media/grey-heron-workflow.png) · [Photo and model credits](CREDITS.md)

## Pretrained models and adaptation

| Stage | Starting model | Training or adaptation used in V1.1 |
| --- | --- | --- |
| Bird detection | Ultralytics pretrained **YOLO11n** (`yolo11n.pt`) | Fine-tuned for a single bird class using annotated birds, hard negatives, and complex scenes. The released application uses the frozen fourth-version detector. |
| Bird vs. background verification | ImageNet-pretrained **YOLO11n-cls** (`yolo11n-cls.pt`) | Fine-tuned as a binary verifier on 6,270 bird crops and 5,778 background crops. This verifier screens detection candidates. |
| Species recognition | Imageomics **BioCLIP 2** (ViT-L/14) and **BioCLIP 2.5** (ViT-H/14) | Both image encoders are frozen. The BioCLIP 2 branch uses an adapted classifier and review rules developed with 736 support photos across 80 species. The BioCLIP 2.5 branch combines taxonomy text scores with a weighted ridge head fitted on 1,434 usable crops across 80 species. These support sets are oriented toward Chinese birds and draw on public iNaturalist observations. |

The default species result blends the two calibrated branch distributions with equal weights. Each accepted detection is cropped from the original image with 1.2× context. The adaptation is in the classifier heads, review rules, and probability fusion. The image encoders retain their pretrained weights.

## AI OpenBuild contribution

The November 6 Windows release integrates a selectable second pass using overlapping 640-pixel tiles at a 480-pixel stride. It checks tile-edge candidates and merges duplicates while preserving the full-image detections. The release also adds a detector handoff that ties initial boxes to the original image and detector configuration. The continuation can reuse those boxes and skip the repeated initial pass. The handoff was tested on one Windows GPU PC.

## Measured results

| Check | Result |
| --- | --- |
| Historical full-image species benchmark | **172/188** correct first choices across 95 species. The prior classifier scored **169/188**. Images with no detection count as incorrect. |
| November 6 regression on previously used photos | **197/197** completed with the original detections preserved. Both modes matched **69/70** annotated sky birds. |
| Negative photos | Both modes returned detection boxes in **10/118** photos and confirmed a species in **0/118**. |
| Additional detections from tiling | The tiled pass evaluated **999** windows and accepted **0** additional boxes in this regression. |

The benchmark and the regression answer different questions. The tested species count is 95. The tiling experiment measured preservation of existing detections on those controls. See [release validation data](evidence/validation.json).

## Run the review example

`python review/demo.py` exercises tile coordinates, internal-edge detection, and JSON handoff validation with a synthetic RGB fixture. [The tiling module](review/tiling.py) comes from the released application. The complete desktop app requires its separate Windows runtime and model assets. The images above show saved outputs from real local inference.

**Keywords** bird-recognition, china-birds, yolo11, bioclip, computer-vision, offline-inference, wildlife-ai
