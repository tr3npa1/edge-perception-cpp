# License and third-party notices

## Project code

Copyright (c) 2026 tr3npa1 and edge-perception-cpp contributors.

Unless a file carries a separate notice, the original source code, build scripts
and documentation in this repository are licensed under the **GNU Affero General
Public License, version 3 only (AGPL-3.0-only)**. See [LICENSE](LICENSE).
The original C++ runtime has the additional NVIDIA linking permission in
[LICENSE-EXCEPTION](LICENSE-EXCEPTION). Preserve both files when redistributing it.
The complete project source is available at
<https://github.com/tr3npa1/edge-perception-cpp>.

## Separately licensed dependencies

This project does not relicense dependencies. Their copyright, attribution,
redistribution and other requirements continue to apply. In particular, an
open-source project license does not make the NVIDIA runtime stack open source.

| Component | Upstream terms / notices |
|---|---|
| Ultralytics YOLO and THOP Python tooling | [Ultralytics AGPL-3.0 / Enterprise licensing](https://www.ultralytics.com/license); upstream copyright and license files remain with the installed packages. No Enterprise license is claimed. |
| ONNX Runtime | [MIT license and third-party notices](https://github.com/microsoft/onnxruntime); the image preserves `/opt/onnxruntime/LICENSE` and `/opt/onnxruntime/ThirdPartyNotices.txt`. |
| OpenCV | [OpenCV licensing](https://opencv.org/license/); distribution-specific dependency notices remain in `/usr/share/doc/`. |
| ONNX and Python build/test packages | Each package retains its own license. Exact top-level requirements are in `requirements*.txt`; installed package license metadata is authoritative for the included version. |
| NVIDIA TensorRT/CUDA/cuDNN and NGC base components | The base image's `/workspace/license.txt` and `/opt/nvidia/entrypoint.d/30-container-license.txt` identify the governing [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and [AI Product-Specific Terms](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/). Preserve component notices and comply with those terms. |
| Ubuntu/system packages and media codecs | Component-specific copyright/license files are preserved under `/usr/share/doc/` and their upstream installation locations. This project grants no additional codec patent rights. |

The Docker image adds the project's C++ inference application to the NVIDIA base;
it is not a redistribution of an unmodified standalone NVIDIA product. There is
no NVIDIA sponsorship, endorsement, safety certification or warranty claim.
The application's corresponding source and build files are included at
`/usr/share/edge-perception/source/`; the OCI source/revision labels identify the
public source release. The image does not include trained models or datasets.

## Dataset and example images

BDD100K imagery, including the three annotated examples under `docs/results/`,
is **not covered by the project's AGPL license**. The underlying images retain
the Regents of the University of California copyright and BDD100K data terms.
The complete upstream notice is reproduced in [licenses/BDD100K.txt](licenses/BDD100K.txt).
These examples are included for research/educational documentation; this project
does not grant unrestricted commercial rights to the imagery.

Full datasets, checkpoints, ONNX models and TensorRT engines are not distributed
in this repository or image. If you separately distribute trained/exported
models, review the applicable Ultralytics/model and dataset terms; the project's
code license is not a substitute for those permissions.
