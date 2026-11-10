# Credits

- Little Grebe photograph by iNaturalist user hurlbert. The recorded photo license is CC0 1.0. [Photo 444658310](https://www.inaturalist.org/photos/444658310) · [Observation 248813120](https://www.inaturalist.org/observations/248813120)
- Grey Heron photograph by iNaturalist contributor 10838672. The recorded photo license is CC0 1.0. [Photo 754004838](https://www.inaturalist.org/photos/754004838) · [Observation 410021035](https://www.inaturalist.org/observations/410021035)
- [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)
- The photos embedded in the recognition-pipeline SVGs use the Little Grebe example above. The frozen V1.1 application was run again on both credited photos on November 10, 2026. The annotated previews come from the application's saved outputs; the crop PNGs are the exact saved classifier inputs. Original and annotated display previews are JPEG re-encodings with metadata removed. No detection boxes were redrawn. See the [fresh inference record](evidence/recognition-example.json) and [display assets](media/recognition-example/). These previously used functional examples are separate from the historical accuracy benchmark.
- Detection and verification base models. [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) · [Ultralytics license information](https://www.ultralytics.com/license)
- Pretrained species encoders. [Imageomics BioCLIP 2](https://huggingface.co/imageomics/bioclip-2) · [Imageomics BioCLIP 2.5 ViT-H/14](https://huggingface.co/imageomics/bioclip-2.5-vith14)
- Historical detector training records list CUB-200-2011, COCO, FBD-SV-2024, IBERBIRDS, and web hard negatives. Species-head adaptation and evaluation used separate iNaturalist photo pools focused on China.
- The complete Windows app uses [PyTorch](https://github.com/pytorch/pytorch), [OpenCLIP](https://github.com/mlfoundations/open_clip), Pillow, and Tkinter. Their upstream terms apply to their respective components.
- The project used OpenAI Codex for AI-assisted implementation, testing support, and documentation.
