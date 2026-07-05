# edge-perception-cpp

Performance-oriented object-detection deployment pipeline using **YOLO26M**, **BDD100K**, **MLflow**, **ONNX**, **ONNX Runtime**, **TensorRT**, **OpenCV**, and native **C++**.

This project trains a YOLO26M detector on BDD100K driving-scene data, exports the trained checkpoint into deployment artifacts, evaluates all available artifacts, and builds toward a native C++ inference pipeline focused on low end-to-end latency.

The goal is not only to train a detector. The goal is to control the full deployment path:

```text
dataset download
dataset preparation
training
MLflow tracking
evaluation
model export
artifact validation
preprocessing
inference
postprocessing
benchmarking
backend comparison
```

---

## Current Project Status

```text
[done]    data/download.py
[done]    data/dataset.py
[done]    training/train_detector.py
[done]    training/export_model.py
[done]    training/evaluate.py
[done]    requirements.txt
[done]    README.md
[done]    .gitignore

[pending] tests/test_parity.py

[pending] CMakeLists.txt
[pending] include/image_processor.hpp
[pending] src/image_processor.cpp
[pending] include/inference_engine.hpp
[pending] src/inference_engine.cpp
[pending] include/postprocess.hpp
[pending] src/postprocess.cpp
[pending] include/benchmark.hpp
[pending] src/benchmark.cpp
[pending] src/main.cpp
[pending] tests/test_inference.cpp

[pending] native C++ ONNX Runtime inference
[pending] optimized OpenCV preprocessing
[pending] optimized C++ YOLO postprocessing
[pending] backend benchmarking
[pending] Docker
```

The Python training/export/evaluation side is complete enough to reproduce the saved workflow. The remaining work is mainly the native C++ deployment and benchmarking pipeline.

---

## Pipeline

```text
BDD100K YOLO dataset
→ data/download.py
→ data/dataset.py
→ training/train_detector.py
→ MLflow tracking
→ training/evaluate.py
→ training/export_model.py
→ exported ONNX / TensorRT artifacts
→ artifact evaluation
→ native C++ inference
→ optimized preprocessing / postprocessing
→ backend benchmarking
```

---

## Repository Structure

```text
edge-perception-cpp/
├── data/
│   ├── download.py
│   ├── dataset.py
│   └── processed/
│       └── bdd100k_yolo/
│           ├── bdd100k.yaml
│           ├── bdd100k_calib_train_as_val.yaml
│           └── dataset_summary.json
├── training/
│   ├── train_detector.py
│   ├── export_model.py
│   └── evaluate.py
├── models/
│   ├── onnx/
│   │   ├── yolo26m_bdd100k_fp32.onnx
│   │   ├── yolo26m_bdd100k_fp16.onnx
│   │   └── yolo26m_bdd100k_int8.onnx
│   └── engine/
│       ├── yolo26m_bdd100k_fp32.engine
│       ├── yolo26m_bdd100k_fp16.engine
│       ├── yolo26m_bdd100k_int8.engine
│       └── yolo26m_bdd100k_int8_traincalib.engine
├── runs/
│   ├── detect/
│   └── evaluate/
├── logs/
├── mlruns/
├── include/
├── src/
├── tests/
├── mlflow.db
├── requirements.txt
├── README.md
└── .gitignore
```

Large generated artifacts are usually not committed to Git, but the structure above shows where the completed local/server outputs are expected after running the full pipeline.

---

## Environment

Recommended Python version:

```text
Python 3.11.x
```

This project is GPU/CUDA-oriented. The training/export workflow was designed for an NVIDIA GPU environment.

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main `requirements.txt` is intentionally a reliable CUDA-focused Python environment. It installs PyTorch CUDA wheels and ONNX Runtime GPU, but it does **not** install TensorRT Python bindings by default.

Verify the local Python environment:

```powershell
python -c "import sys, torch, torchvision, ultralytics, mlflow, onnx, onnxruntime as ort, yaml; print('Python:', sys.version); print('Torch:', torch.__version__); print('TorchVision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print('Ultralytics:', ultralytics.__version__); print('MLflow:', mlflow.__version__); print('ONNX:', onnx.__version__); print('ORT:', ort.__version__); print('ORT providers:', ort.get_available_providers())"

python -m pip check
```

Expected core versions for the final local/server-like environment:

```text
Python:       3.11.x
Torch:        2.12.1+cu130
TorchVision:  0.27.1+cu130
Ultralytics:  8.4.87
MLflow:       3.14.0
ONNX:         1.22.0
ONNX Runtime: 1.27.0
PyYAML:       6.0.3
```

`CUDA available: True` is expected on an NVIDIA GPU machine with compatible drivers. ONNX Runtime provider availability may vary by local Windows DLL/runtime setup.

---

## Requirements

Use a pinned `requirements.txt` for reproducibility rather than broad `>=` constraints.

Recommended `requirements.txt`:

```text
# edge-perception-cpp reproducible GPU-focused Python environment
# Recommended Python: 3.11.x
#
# Install:
#   py -3.11 -m venv .venv
#   .\.venv\Scripts\Activate.ps1
#   python -m pip install --upgrade pip
#   python -m pip install -r requirements.txt
#
# Notes:
# - This project is GPU/deployment focused.
# - PyTorch is pinned to CUDA 13.0 wheels.
# - ONNX Runtime GPU is used instead of CPU-only onnxruntime.
# - TensorRT Python bindings are intentionally NOT included here because
#   TensorRT pip dependencies can be platform/runtime fragile.
# - Native TensorRT .engine export should be done on a properly configured
#   NVIDIA GPU server/runtime.

--extra-index-url https://download.pytorch.org/whl/cu130

# Packaging
setuptools==81.0.0
wheel==0.47.0

# Dataset download
kaggle==2.2.3

# YOLO training/export/evaluation
ultralytics==8.4.87
ultralytics-thop==2.0.20

# Core ML - CUDA 13.0 PyTorch wheels
torch==2.12.1+cu130
torchvision==0.27.1+cu130

# Experiment tracking
mlflow==3.14.0

# ONNX export and Python-side inference
onnx==1.22.0
onnxslim==0.1.94

# ONNX Runtime GPU package.
# This replaces CPU-only onnxruntime.
onnxruntime-gpu==1.27.0

# Image processing and data utilities
opencv-python==5.0.0.93
numpy==2.2.6
PyYAML==6.0.3

# Testing
pytest==9.1.1
```

If `onnxruntime-gpu` causes local Windows provider/DLL issues, replace it with CPU ONNX Runtime:

```powershell
python -m pip uninstall -y onnxruntime-gpu
python -m pip install onnxruntime==1.27.0
```

The CPU package is enough for local MLflow viewing, code cleanup, and basic ONNX inspection.

TensorRT is intentionally optional and should not be placed in the main requirements file. Keeping it out makes a fresh user install much less likely to fail.

---

## Dataset

The project uses a YOLO-format BDD100K dataset hosted on Kaggle.

Expected raw dataset location:

```text
data/raw/bdd100k/bdd100k/
```

Expected raw layout:

```text
data/raw/bdd100k/bdd100k/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Expected classes:

```text
person
rider
car
bus
truck
bike
motor
traffic light
traffic sign
train
```

Large raw dataset files are not committed to Git.

---

## Dataset Download

Dataset download is handled by:

```text
data/download.py
```

The script downloads the Kaggle dataset, extracts it, performs a raw dataset smoke check, and writes a download manifest.

Run the default download:

```bash
python data/download.py
```

Run a clean re-download:

```bash
python data/download.py --clean --force-download
```

Validate existing downloaded data:

```bash
python data/download.py --validate-only
```

Expected raw output:

```text
data/raw/bdd100k/
├── kaggle_download/
├── bdd100k/
└── download_manifest.json
```

---

## Dataset Preparation

Dataset preparation is handled by:

```text
data/dataset.py
```

The raw Kaggle dataset is preserved unchanged. The preparation script validates the YOLO dataset and writes a clean local training YAML.

Run:

```bash
python data/dataset.py
```

Validate only:

```bash
python data/dataset.py --validate-only
```

Generated files:

```text
data/processed/bdd100k_yolo/bdd100k.yaml
data/processed/bdd100k_yolo/dataset_summary.json
```

The training, export, and evaluation scripts use:

```text
data/processed/bdd100k_yolo/bdd100k.yaml
```

The preparation script checks:

```text
source dataset YAML
class names
nc value
train / val / test image paths
matching YOLO label files
YOLO label line format
```

---

## Model

Detector:

```text
YOLO26M
```

Training target:

```text
dataset: BDD100K
input size: 640x640
batch size: configurable
device: single NVIDIA GPU
precision: AMP enabled through Ultralytics
```

Deployment/export target:

```text
batch size: 1
input size: 640x640
ONNX FP32
ONNX FP16
ONNX INT8
TensorRT engine FP32
TensorRT engine FP16
TensorRT engine INT8
```

---

## MLflow Tracking

MLflow is used for training, export, and evaluation tracking.

Start the local MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

If using PowerShell on Windows:

```powershell
python -m mlflow ui `
  --backend-store-uri "sqlite:///E:/tmkc/edge-perception-cpp/mlflow.db" `
  --host 127.0.0.1 `
  --port 5000
```

If a copied `mlflow.db` was created by an older MLflow version, back it up and upgrade the local database schema:

```powershell
Copy-Item "mlflow.db" "mlflow.db.backup-before-local-upgrade" -Force
python -m mlflow db upgrade "sqlite:///E:/tmkc/edge-perception-cpp/mlflow.db"
```

Do not upgrade or upload a migrated database back to an older server environment unless that server uses the same MLflow version.

---

## Training

Training is handled by:

```text
training/train_detector.py
```

Expected training input:

```text
data/processed/bdd100k_yolo/bdd100k.yaml
```

Training responsibilities:

```text
load YOLO26M weights
validate the prepared BDD100K YAML
train or resume YOLO26M
save best.pt and last.pt
save local training metadata
log the run to MLflow
log training configuration and artifacts
```

Full training command:

```bash
python training/train_detector.py \
  --mlflow-tracking-uri sqlite:///mlflow.db \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --device 0
```

Smoke-test command:

```bash
python training/train_detector.py \
  --mlflow-tracking-uri sqlite:///mlflow.db \
  --epochs 1 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --fraction 0.05 \
  --name smoke_yolo26m_bdd100k \
  --mlflow-run-name smoke_yolo26m_bdd100k
```

Important output path:

```text
runs/train/yolo26m_bdd100k/weights/best.pt
```

Use `best.pt` for evaluation, export, and deployment.

---

## Checkpoints

Ultralytics saves two main checkpoints:

```text
best.pt
last.pt
```

Use:

```text
best.pt
```

for evaluation, export, and deployment.

Do not blindly deploy:

```text
last.pt
```

because `last.pt` is only the final epoch checkpoint, not necessarily the best validation checkpoint.

---

## Evaluation

Evaluation is handled by:

```text
training/evaluate.py
```

The evaluation script measures detection quality separately from deployment speed.

Responsibilities:

```text
evaluate PyTorch best.pt
evaluate provided ONNX artifacts
evaluate provided native TensorRT engine artifacts
extract mAP50, mAP75, mAP50-95, precision, recall
extract per-class metrics when available
optionally check ONNX Runtime CPU/CUDA/TensorRT providers
optionally run final-detection parity checks
log metrics and JSON summaries to MLflow
continue across artifacts if one backend fails
```

Recommended full artifact evaluation command:

```bash
python training/evaluate.py \
  --weights runs/train/yolo26m_bdd100k/weights/best.pt \
  --data data/processed/bdd100k_yolo/bdd100k.yaml \
  --split test \
  --imgsz 640 \
  --batch 1 \
  --workers 8 \
  --device 0 \
  --name yolo26m_bdd100k_eval_all_artifacts_clean \
  --exist-ok \
  --onnx \
    models/onnx/yolo26m_bdd100k_fp32.onnx \
    models/onnx/yolo26m_bdd100k_fp16.onnx \
    models/onnx/yolo26m_bdd100k_int8.onnx \
  --engine \
    models/engine/yolo26m_bdd100k_fp32.engine \
    models/engine/yolo26m_bdd100k_fp16.engine \
    models/engine/yolo26m_bdd100k_int8.engine \
  --check-ort-cuda \
  --no-check-ort-tensorrt \
  --no-parity \
  --mlflow \
  --mlflow-tracking-uri sqlite:///mlflow.db \
  --mlflow-experiment-name edge-perception-cpp \
  --mlflow-run-name yolo26m_bdd100k_eval_all_artifacts_clean \
  --mlflow-log-artifacts
```

Important default behavior:

```text
batch defaults to 1 for exported static-batch artifacts
parity is disabled by default to avoid false dirty failures
ORT TensorRT checks are optional and disabled by default
optional artifact/backend failures do not block the rest by default
```

No speed claims should be reported from `training/evaluate.py`. Benchmark speed belongs in the C++ benchmarking path.

---

## Model Export

Model export is handled by:

```text
training/export_model.py
```

Expected input checkpoint:

```text
runs/train/yolo26m_bdd100k/weights/best.pt
```

Expected dataset YAML:

```text
data/processed/bdd100k_yolo/bdd100k.yaml
```

Responsibilities:

```text
export ONNX FP32
export ONNX FP16
export ONNX INT8 / PTQ
export native TensorRT engine FP32
export native TensorRT engine FP16
export native TensorRT engine INT8
write per-artifact metadata JSON
write global export summary JSON
inspect ONNX graph inputs/outputs
inspect ONNX Runtime providers when requested
optionally smoke-load TensorRT engine artifacts
continue across variants if one backend fails
```

Recommended full export command:

```bash
python training/export_model.py \
  --weights runs/train/yolo26m_bdd100k/weights/best.pt \
  --data data/processed/bdd100k_yolo/bdd100k.yaml \
  --variants all \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --basename yolo26m_bdd100k \
  --opset 18 \
  --calibration-fraction 1.0 \
  --calibration-max-images 0 \
  --workspace 8 \
  --engine-int8-calibration-split train \
  --overwrite \
  --engine-smoke-load \
  --check-ort-cuda \
  --no-check-ort-tensorrt
```

Expected ONNX outputs:

```text
models/onnx/yolo26m_bdd100k_fp32.onnx
models/onnx/yolo26m_bdd100k_fp16.onnx
models/onnx/yolo26m_bdd100k_int8.onnx
```

Expected native TensorRT engine outputs:

```text
models/engine/yolo26m_bdd100k_fp32.engine
models/engine/yolo26m_bdd100k_fp16.engine
models/engine/yolo26m_bdd100k_int8.engine
models/engine/yolo26m_bdd100k_int8_traincalib.engine
```

Expected metadata outputs:

```text
models/onnx/yolo26m_bdd100k_onnx_fp32_metadata.json
models/onnx/yolo26m_bdd100k_onnx_fp16_metadata.json
models/onnx/yolo26m_bdd100k_onnx_int8_metadata.json
models/engine/yolo26m_bdd100k_engine_fp32_metadata.json
models/engine/yolo26m_bdd100k_engine_fp16_metadata.json
models/engine/yolo26m_bdd100k_engine_int8_metadata.json
models/yolo26m_bdd100k_export_summary.json
```

Important INT8 note:

```text
ONNX INT8 and native TensorRT INT8 are separate deployment artifacts.

The native TensorRT INT8 path should export directly from the PyTorch checkpoint through Ultralytics/TensorRT. It should not try to build a TensorRT engine from the ONNX Runtime QDQ INT8 ONNX graph.

For full-train calibration, the export script may create a temporary YAML where val=train because Ultralytics uses the YAML val split for TensorRT INT8 calibration.
```

---

## ONNX Runtime and TensorRT Notes

The project distinguishes between three deployment routes:

```text
ONNX file + ONNX Runtime CPU Execution Provider
ONNX file + ONNX Runtime CUDA / TensorRT Execution Provider
native TensorRT .engine file
```

These are not the same thing.

The ONNX Runtime TensorRT Execution Provider consumes an ONNX file and may build/cache TensorRT engines internally at runtime.

The native TensorRT `.engine` file is a TensorRT-specific serialized engine. It is usually specific to:

```text
GPU architecture
TensorRT version
CUDA version
operating system
driver/runtime environment
input shape/profile
```

Therefore, `.engine` files exported on a Linux GPU server should be treated as saved deployment artifacts for that environment. Do not expect them to run unchanged on a different Windows machine.

`training/export_model.py` keeps TensorRT Python bindings optional. The TensorRT Python API is imported only inside the optional fallback builder path and is ignored by static type checkers. This prevents local VS Code/Pylance warnings from becoming a required dependency for every user.

---

## Optional TensorRT Setup

TensorRT is **not** installed by the main `requirements.txt`.

This is intentional.

Native TensorRT export and TensorRT engine smoke-loading require a compatible combination of:

```text
NVIDIA GPU
NVIDIA driver
CUDA runtime
TensorRT runtime
operating system
Python bindings
input shape/profile
```

For this project, the safest workflow is:

```text
local PC:
  install requirements.txt
  use MLflow
  inspect ONNX
  edit code
  build docs
  work on C++ source

GPU server:
  train YOLO26M
  export native TensorRT .engine files
  evaluate TensorRT artifacts
  benchmark deployment backends
```

If a machine is already configured for TensorRT, the optional Python fallback path in `training/export_model.py` can use:

```python
import tensorrt as trt
```

but that import is optional and not required for normal local development.

Do not add TensorRT pip packages to the main `requirements.txt` unless the install has been tested cleanly on the target platform. If TensorRT pip installation fails, keep TensorRT documented as an optional environment-specific dependency rather than making the whole project install fail.

---

## Parity Testing

Dedicated parity testing is planned for:

```text
tests/test_parity.py
```

The evaluation script also has optional final-detection parity checks, but they are disabled by default because tiny numeric differences between PyTorch, ONNX, and TensorRT can cause strict final-detection parity to fail even when mAP validation is successful.

Use explicit parity checks only when debugging export correctness:

```bash
python training/evaluate.py \
  --weights runs/train/yolo26m_bdd100k/weights/best.pt \
  --data data/processed/bdd100k_yolo/bdd100k.yaml \
  --onnx models/onnx/yolo26m_bdd100k_fp32.onnx \
  --parity \
  --parity-images 32
```

---

## Native C++ Inference

The native inference pipeline will use:

```text
C++17 or newer
OpenCV
ONNX Runtime C++
```

Planned executable:

```text
edge_perception
```

Core inference flow:

```text
load image
↓
preprocess with OpenCV
↓
write directly into model input buffer
↓
run inference
↓
decode detections
↓
apply confidence threshold
↓
apply NMS if needed
↓
scale boxes back to original image
↓
return final detections
```

The C++ path is designed around reusable components:

```text
ImageProcessor
InferenceEngine
Postprocess
Benchmark
```

---

## Image Preprocessing

OpenCV loads images as:

```text
HWC BGR uint8
```

YOLO-style deployed models typically expect:

```text
NCHW RGB float32
```

The preprocessing implementation must handle:

```text
image load
resize or letterbox
BGR to RGB conversion
uint8 to float conversion
normalization to 0.0-1.0
HWC to NCHW layout conversion
batch dimension creation
letterbox metadata for box rescaling
```

The optimized path should avoid unnecessary intermediate buffers where practical.

---

## Postprocessing

Postprocessing will be handled by:

```text
include/postprocess.hpp
src/postprocess.cpp
```

The postprocessing implementation should support:

```text
confidence thresholding
class IDs
box decoding
NMS if required
no-NMS path if exported model returns final detections
box scaling back to original image size
```

Drawing detections should be optional and excluded from benchmark timing unless explicitly requested.

---

## Benchmarking

Benchmarking will be handled by:

```text
include/benchmark.hpp
src/benchmark.cpp
src/main.cpp
```

Benchmark mode should measure:

```text
image loading latency
preprocessing latency
inference latency
postprocessing latency
end-to-end latency
```

Benchmark rules:

```text
use the same image set across backends
use the same model variant across backends
use the same input size
use the same confidence threshold
use the same IoU threshold
run warmup iterations before timed iterations
do not include model export time
do not include TensorRT engine build time in steady-state inference latency
do not include drawing time unless reported separately
```

Benchmarking should report:

```text
mean latency
median latency
p95 latency
FPS
model size
precision mode
backend
input size
hardware
```

Benchmark results should only be added after they are measured.

---

## Backend Plan

Planned backend support:

```text
ONNX Runtime CPU
ONNX Runtime CUDA Execution Provider
ONNX Runtime TensorRT Execution Provider
native TensorRT backend if needed
```

The project should distinguish between:

```text
raw model inference latency
end-to-end pipeline latency
```

Raw TensorRT engine latency is treated as a strong lower-bound model-execution baseline. The native C++ pipeline focuses on reducing full end-to-end deployment latency.

---

## Artifact Policy

The repository tracks source code, build configuration, and documentation.

The repository should not track large raw or generated artifacts by default:

```text
data/raw/
.venv/
runs/
mlruns/
models/
logs/
weights/
checkpoints/
outputs/
benchmarks/
*.pt
*.pth
*.onnx
*.engine
*.db
*.tar.gz
*.zip
```

Small processed metadata/config files may be useful to preserve or document separately:

```text
data/processed/bdd100k_yolo/bdd100k.yaml
data/processed/bdd100k_yolo/dataset_summary.json
```

If large artifacts need to be preserved, store them outside Git, for example in a release asset, external drive, cloud storage, or a manually downloaded results archive.

Recommended archive contents after a completed server run:

```text
training/
models/
runs/
logs/
mlruns/
data/processed/
mlflow.db
requirements.txt
yolo26m.pt
yolo26n.pt
```

Do not include:

```text
data/raw/
.venv/
```

---

## Full Reproduction Flow

A fresh run should follow this order:

```text
1. create Python 3.11 virtual environment
2. install requirements
3. run data/download.py
4. run data/dataset.py
5. start or configure MLflow
6. run training/train_detector.py
7. run training/evaluate.py on best.pt
8. run training/export_model.py
9. run training/evaluate.py on exported ONNX / engine artifacts
10. inspect MLflow, logs, and JSON summaries
11. proceed to native C++ inference and benchmarking
```

---

## Development Order

```text
1. data/download.py
2. README.md
3. .gitignore
4. requirements.txt
5. data/dataset.py
6. training/train_detector.py
7. training/evaluate.py
8. training/export_model.py
9. tests/test_parity.py
10. CMakeLists.txt
11. include/image_processor.hpp
12. src/image_processor.cpp
13. include/inference_engine.hpp
14. src/inference_engine.cpp
15. include/postprocess.hpp
16. src/postprocess.cpp
17. include/benchmark.hpp
18. src/benchmark.cpp
19. src/main.cpp
20. tests/test_inference.cpp
21. CUDA / TensorRT provider support
22. FP16 export and benchmark support
23. INT8 calibration
24. async / pipelined benchmark mode
25. Dockerfile
26. final README polish
```

---

## Current Focus

Completed Python-side pipeline:

```text
Kaggle BDD100K YOLO dataset
→ data/download.py
→ data/raw/bdd100k/bdd100k/
→ data/dataset.py
→ data/processed/bdd100k_yolo/bdd100k.yaml
→ training/train_detector.py
→ runs/train/yolo26m_bdd100k/weights/best.pt
→ training/export_model.py
→ models/onnx/*.onnx and models/engine/*.engine
→ training/evaluate.py
→ MLflow + local JSON summaries
```

Next focus:

```text
clean repository directory
commit finalized Python training/export/evaluation files
start native C++ inference path
implement optimized preprocessing
implement YOLO postprocessing
benchmark CPU / CUDA / TensorRT deployment backends
```
