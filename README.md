# edge-perception-cpp

Performance-oriented object detection deployment pipeline using YOLO26M, ONNX Runtime, OpenCV, and native C++.

This project trains a YOLO26M detector on BDD100K driving-scene data, exports the trained model for deployment, and implements a native C++ inference pipeline focused on low end-to-end latency.

The goal is to control the full deployment path:

```text
dataset preparation
training
evaluation
model export
preprocessing
inference
postprocessing
benchmarking
backend comparison
```

---

## Pipeline

```text
BDD100K YOLO dataset
→ dataset download
→ dataset validation and local YAML preparation
→ YOLO26M training
→ MLflow experiment tracking
→ evaluation
→ ONNX / TensorRT export
→ parity testing
→ native C++ inference
→ optimized OpenCV preprocessing
→ optimized YOLO postprocessing
→ backend benchmarking
→ optional FP16 / INT8 / TensorRT acceleration
```

---

## Project Status

```text
[done] data/download.py
[done] README.md
[done] .gitignore
[done] requirements.txt
[done] data/dataset.py
[done] training/train_detector.py

[pending] training/evaluate.py
[pending] training/export_model.py
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

[pending] CUDA / TensorRT provider support
[pending] FP16 export and benchmark support
[pending] INT8 calibration
[pending] async / pipelined benchmark mode
[pending] Docker
```

---

## Repository Structure

```text
edge-perception-cpp/
├── data/
│   ├── download.py
│   └── dataset.py
├── training/
│   ├── train_detector.py
│   ├── evaluate.py
│   └── export_model.py
├── include/
│   ├── image_processor.hpp
│   ├── inference_engine.hpp
│   ├── postprocess.hpp
│   └── benchmark.hpp
├── src/
│   ├── image_processor.cpp
│   ├── inference_engine.cpp
│   ├── postprocess.cpp
│   ├── benchmark.cpp
│   └── main.cpp
├── tests/
│   ├── test_parity.py
│   └── test_inference.cpp
├── CMakeLists.txt
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

The project uses a YOLO-format BDD100K dataset hosted on Kaggle.

Expected raw dataset location:

```text
data/raw/bdd100k/bdd100k/
```

Expected raw dataset layout:

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

Large dataset files are not committed to Git.

---

## Dataset Download

Dataset download is handled by:

```text
data/download.py
```

The script downloads the Kaggle dataset, extracts it, performs a raw dataset smoke check, and writes a download manifest.

Install the Kaggle CLI:

```bash
pip install kaggle
```

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

Training scripts use:

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

Planned detector:

```text
YOLO26M
```

Initial training target:

```text
dataset: BDD100K
input size: 640x640
batch size: 16
epochs: 100
device: single GPU
precision: AMP enabled
```

Initial deployment target:

```text
batch size: 1
input size: 640x640
precision: FP32
```

Later deployment targets:

```text
FP16
INT8
TensorRT engine
ONNX Runtime CUDA provider
ONNX Runtime TensorRT provider
```

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

The training script is responsible for:

```text
loading YOLO26M weights
validating the prepared BDD100K YAML
training or resuming YOLO26M
saving best.pt and last.pt
saving local training metadata
logging the run to MLflow
logging training configuration and artifacts
```

Full training command:

```bash
python training/train_detector.py \
  --mlflow-tracking-uri http://127.0.0.1:5000 \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

Smoke-test command:

```bash
python training/train_detector.py \
  --mlflow-tracking-uri http://127.0.0.1:5000 \
  --epochs 1 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --fraction 0.05 \
  --name smoke_yolo26m_bdd100k \
  --mlflow-run-name smoke_yolo26m_bdd100k
```

The checkpoint to use for later evaluation and export is:

```text
runs/train/<run_name>/weights/best.pt
```

The final epoch checkpoint is:

```text
runs/train/<run_name>/weights/last.pt
```

---

## MLflow Tracking

MLflow is used for experiment tracking during training.

The training script logs:

```text
training parameters
dataset information
training metadata
Ultralytics args.yaml
Ultralytics results.csv
training duration
run directory
best checkpoint path
last checkpoint path
```

The Ultralytics run directory also stores:

```text
results.csv
results.png
args.yaml
weights/best.pt
weights/last.pt
```

MLflow is useful for comparing runs, while the local Ultralytics run folder remains the source of truth for training outputs.

Start an MLflow server separately if needed:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then train with:

```bash
python training/train_detector.py --mlflow-tracking-uri http://127.0.0.1:5000
```

---

## Checkpoints

Ultralytics saves two main checkpoints:

```text
best.pt
last.pt
```

Use `best.pt` for evaluation, export, and deployment.

```text
best.pt
    best validation-fitness checkpoint

last.pt
    final epoch checkpoint
```

The deployment pipeline should not blindly use `last.pt`.

---

## Evaluation

Evaluation will be handled by:

```text
training/evaluate.py
```

The evaluation script will measure detection quality separately from deployment speed.

Planned outputs:

```text
mAP50
mAP50-95
precision
recall
per-class metrics
evaluation metrics JSON
sample predictions
```

No speed claims should be reported from the evaluation script.

---

## Model Export

Model export will be handled by:

```text
training/export_model.py
```

The export script will support deployment-focused model formats.

Planned exports:

```text
FP32 ONNX
FP16 ONNX
TensorRT engine
INT8 engine or quantized model
```

The export script should record:

```text
exported model path
input tensor name
input tensor shape
output tensor names
output tensor shapes
precision mode
opset
backend/export format
smoke-test result
export metadata
```

---

## Parity Testing

Parity testing will be handled by:

```text
tests/test_parity.py
```

The test will compare outputs between the trained model and exported deployment models.

Planned comparisons:

```text
PyTorch / Ultralytics output
ONNX Runtime output
TensorRT output if available
```

The goal is to confirm that exported models remain numerically and functionally consistent before C++ benchmarking.

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

## Performance Design

Performance is a first-class project goal.

The native pipeline should minimize:

```text
unnecessary memory allocation
unnecessary image copies
Python/framework overhead
slow generic preprocessing
slow postprocessing
unmeasured visualization overhead
```

Planned implementation choices:

```text
preallocated input buffers
preallocated output buffers where practical
single-pass BGR HWC uint8 to RGB NCHW float conversion
static input shapes
warmup iterations before timing
separate timing for preprocessing, inference, postprocessing, and total latency
backend selection through the inference engine
optional CUDA and TensorRT acceleration
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

Benchmark results should only be added after they are measured locally.

---

## Backends

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
normalization
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

## Artifact Policy

The repository tracks source code, build configuration, and documentation.

The repository does not track large or generated artifacts such as:

```text
data/raw/
data/processed/
runs/
mlruns/
models/
weights/
checkpoints/
outputs/
benchmarks/
*.pt
*.pth
*.onnx
*.engine
```

---

## Current Focus

Completed files:

```text
data/download.py
README.md
.gitignore
requirements.txt
data/dataset.py
training/train_detector.py
```

Current training flow:

```text
Kaggle BDD100K YOLO dataset
→ data/download.py
→ data/raw/bdd100k/bdd100k/
→ data/dataset.py
→ data/processed/bdd100k_yolo/bdd100k.yaml
→ training/train_detector.py
→ runs/train/<run_name>/weights/best.pt
```

Next file:

```text
training/evaluate.py
```