# edge-perception-cpp

High-performance object detection deployment project for automotive-style perception workloads.

This project trains and exports a YOLO26M object detector on BDD100K, evaluates deployment artifacts across PyTorch, ONNX Runtime, and native TensorRT, and prepares the repository for a C++ inference backend using ONNX Runtime / TensorRT / OpenCV / CMake.

The goal is not only to train a detector, but to build a production-style deployment pipeline:

```text
dataset preparation
↓
YOLO training
↓
artifact export
↓
artifact validation
↓
backend comparison
↓
native C++ inference engine
↓
end-to-end benchmark
```

---

## Current Status

Completed:

```text
BDD100K YOLO-format dataset pipeline
YOLO26M training
MLflow training/evaluation tracking
ONNX FP32 export
ONNX FP16 export
ONNX all-Conv INT8 QDQ export
native TensorRT FP32 engine export
native TensorRT FP16 engine export
native TensorRT INT8 engine export
artifact validation on BDD100K test split
ONNX Runtime CPU/CUDA session checks
final backend quality matrix
```

In progress / next:

```text
C++ inference backend
OpenCV preprocessing
ONNX Runtime C++ inference
TensorRT runtime path
postprocessing and output formatting
CMake build system
C++ benchmark harness
end-to-end latency reporting
```

Important: Python validation speed numbers in this README are **not final deployment latency**. Final end-to-end latency should be reported only after the C++ backend measures preprocessing, inference, postprocessing, and output handling in one controlled benchmark.

---

## Dataset

Dataset:

```text
BDD100K
```

Prepared dataset YAML:

```text
data/processed/bdd100k_yolo/bdd100k.yaml
```

Test split used for final evaluation:

```text
images: 20,000
instances: 367,727
classes: 10
```

Classes:

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

---

## Final Evaluation Summary

Final evaluation run:

```text
runs/evaluate/yolo26m_bdd100k_eval_final_allconv_int82
```

MLflow run id:

```text
d8010b651ef44d26a074612c0df1e08f
```

Evaluation settings:

```text
split: test
imgsz: 640
batch: 1
confidence threshold: 0.001
IoU threshold: 0.7
max detections: 300
parity: disabled
ORT TensorRT EP check: disabled
```

Final quality matrix:

| Backend / Artifact | Precision | Recall | mAP50 | mAP50-95 | Validation Inference ms/img |
|---|---:|---:|---:|---:|---:|
| PyTorch `best.pt` | 0.756 | 0.529 | 0.602 | 0.341 | 27.65 |
| ONNX FP32 | 0.731 | 0.544 | 0.601 | 0.340 | 38.59 |
| ONNX FP16 | 0.731 | 0.544 | 0.600 | 0.340 | 15.01 |
| ONNX all-Conv INT8 QDQ | 0.729 | 0.490 | 0.538 | 0.286 | 57.03 |
| TensorRT FP32 engine | 0.731 | 0.544 | 0.601 | 0.340 | 23.13 |
| TensorRT FP16 engine | 0.731 | 0.544 | 0.600 | 0.340 | 6.58 |
| TensorRT INT8 engine | 0.828 | 0.401 | 0.564 | 0.311 | 4.66 |

Interpretation:

```text
PyTorch, ONNX FP32, ONNX FP16, TensorRT FP32, and TensorRT FP16 preserve almost identical mAP.
TensorRT FP16 gives the best accuracy-preserving accelerated engine result.
TensorRT INT8 is the fastest validated engine artifact but trades recall for precision.
ONNX all-Conv INT8 is valid and non-zero, but loses more mAP than FP16 and is not the final speed path.
```

The speed column above comes from Ultralytics/Python validation and should not be treated as final deployment latency.

---

## INT8 Quantization Notes

This project tested two ONNX INT8 strategies.

### 1. Naive ONNX INT8 QDQ

A naive static ONNX INT8 QDQ export was attempted first.

That artifact was rejected because it corrupted the final detection output tensor and produced zero mAP.

The failure mode was visible in the raw model output:

```text
expected output row:
[x1, y1, x2, y2, confidence, class_id]

broken INT8 output:
[0, 0, 8.69, 8.69, 0, 2.897983]
```

The model still ran, but the numerical output was unusable.

### 2. Final ONNX all-Conv INT8 QDQ

The final ONNX INT8 artifact uses all-Conv QDQ quantization:

```text
models/onnx/yolo26m_bdd100k_allconv_int8.onnx
```

The final all-Conv INT8 export quantizes every Conv node that ONNX Runtime quantizes:

```text
FP32 Conv nodes: 112
INT8 Conv nodes: 112
QDQ-quantized Conv nodes: 112
FP32-like / unquantized Conv nodes: 0
```

It intentionally does not quantize the whole graph. Non-Conv output-selection and detection-formatting logic remains floating point.

This distinction matters because the final exported YOLO output has shape:

```text
[1, 300, 6]
```

Each row is:

```text
[x1, y1, x2, y2, confidence, class_id]
```

Those columns have very different numeric ranges. Quantizing the wrong final-output path can destroy confidence values and class IDs. The all-Conv strategy avoids the broken full-graph behavior while still quantizing the convolutional compute graph.

Final ONNX all-Conv INT8 result:

```text
mAP50:     0.538
mAP50-95:  0.286
```

This is a valid artifact, but TensorRT INT8 remains the better INT8 deployment path in this project.

---

## Final Artifacts

The useful final ONNX artifacts are:

```text
models/onnx/yolo26m_bdd100k_fp32.onnx
models/onnx/yolo26m_bdd100k_fp16.onnx
models/onnx/yolo26m_bdd100k_allconv_int8.onnx
```

The useful final native TensorRT artifacts are:

```text
models/engine/yolo26m_bdd100k_fp32.engine
models/engine/yolo26m_bdd100k_fp16.engine
models/engine/yolo26m_bdd100k_int8.engine
```

These files are large generated artifacts and are not intended to be committed to Git.

---

## Repository Structure

```text
edge-perception-cpp/
├── data/
│   ├── download.py
│   └── dataset.py
├── training/
│   ├── train_detector.py
│   ├── export_model.py
│   └── evaluate.py
├── docs/
│   └── backend_matrix.md
├── models/
│   ├── onnx/
│   └── engine/
├── runs/
├── requirements.txt
├── requirements-tensorrt.txt
├── .gitignore
└── README.md
```

Planned C++ layout:

```text
include/
├── image_processor.hpp
├── inference_engine.hpp
├── postprocess.hpp
└── benchmark.hpp

src/
├── image_processor.cpp
├── inference_engine.cpp
├── postprocess.cpp
├── benchmark.cpp
└── main.cpp

tests/
├── test_parity.py
└── test_inference.cpp

CMakeLists.txt
```

---

## Environment Setup

Recommended Python version:

```text
Python 3.11.x
```

Create and activate a virtual environment on Windows PowerShell:

```powershell
cd E:\tmkc\edge-perception-cpp

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorRT support is optional and isolated:

```powershell
python -m pip install -r requirements-tensorrt.txt
```

TensorRT Python packages and DLL visibility can be fragile on Windows. Native TensorRT engine export/evaluation requires a compatible CUDA, cuDNN, TensorRT, GPU driver, and Python package stack.

---

## MLflow

Start MLflow locally:

```powershell
cd E:\tmkc\edge-perception-cpp
.\.venv\Scripts\Activate.ps1

mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

MLflow is used for:

```text
training metrics
evaluation metrics
backend comparison runs
artifact reports
JSON summaries
```

`mlflow.db` and `mlruns/` are generated local tracking artifacts and are not committed.

---

## Training

Training entrypoint:

```text
training/train_detector.py
```

Expected trained checkpoint:

```text
runs/train/yolo26m_bdd100k/weights/best.pt
```

This project uses the trained checkpoint as the source for all exports and evaluations.

---

## Export

Export entrypoint:

```text
training/export_model.py
```

Main export responsibilities:

```text
export ONNX FP32
export ONNX FP16
export ONNX all-Conv INT8 QDQ
export native TensorRT FP32 engine
export native TensorRT FP16 engine
export native TensorRT INT8 engine
write metadata JSON
inspect ONNX graph inputs/outputs
optionally check ONNX Runtime CPU/CUDA readiness
optionally smoke-load TensorRT engines
```

Recommended full export command for FP32/FP16 ONNX and TensorRT engines:

```powershell
python training\export_model.py `
    --weights runs\train\yolo26m_bdd100k\weights\best.pt `
    --data data\processed\bdd100k_yolo\bdd100k.yaml `
    --variants onnx_fp32 onnx_fp16 engine_fp32 engine_fp16 engine_int8 `
    --imgsz 640 `
    --batch 1 `
    --device 0 `
    --basename yolo26m_bdd100k `
    --opset 18 `
    --calibration-fraction 1.0 `
    --calibration-max-images 0 `
    --workspace 8 `
    --engine-int8-calibration-split train `
    --overwrite `
    --engine-smoke-load `
    --check-ort-cpu `
    --check-ort-cuda `
    --no-check-ort-tensorrt
```

Recommended all-Conv ONNX INT8 export command:

```powershell
Copy-Item `
    "models\onnx\yolo26m_bdd100k_fp32.onnx" `
    "models\onnx\yolo26m_bdd100k_allconv_fp32.onnx" `
    -Force

python training\export_model.py `
    --weights runs\train\yolo26m_bdd100k\weights\best.pt `
    --data data\processed\bdd100k_yolo\bdd100k.yaml `
    --variants onnx_int8 `
    --imgsz 640 `
    --batch 1 `
    --device 0 `
    --basename yolo26m_bdd100k_allconv `
    --calibration-fraction 1.0 `
    --calibration-max-images 0 `
    --overwrite `
    --check-ort-cpu `
    --no-check-ort-cuda `
    --no-check-ort-tensorrt `
    --ort-run-dummy
```

Temporary helper file:

```text
models/onnx/yolo26m_bdd100k_allconv_fp32.onnx
```

This file is only a basename helper for the INT8 export and can be deleted after `yolo26m_bdd100k_allconv_int8.onnx` is produced.

---

## Evaluation

Evaluation entrypoint:

```text
training/evaluate.py
```

Final evaluation command:

```powershell
python training\evaluate.py `
    --weights runs\train\yolo26m_bdd100k\weights\best.pt `
    --data data\processed\bdd100k_yolo\bdd100k.yaml `
    --split test `
    --imgsz 640 `
    --batch 1 `
    --workers 8 `
    --device 0 `
    --output runs\evaluate `
    --name yolo26m_bdd100k_eval_final_allconv_int8 `
    --conf 0.001 `
    --iou 0.7 `
    --max-det 300 `
    --plots `
    --save-json `
    --eval-pytorch `
    --eval-artifacts `
    --onnx `
        models\onnx\yolo26m_bdd100k_fp32.onnx `
        models\onnx\yolo26m_bdd100k_fp16.onnx `
        models\onnx\yolo26m_bdd100k_allconv_int8.onnx `
    --engine `
        models\engine\yolo26m_bdd100k_fp32.engine `
        models\engine\yolo26m_bdd100k_fp16.engine `
        models\engine\yolo26m_bdd100k_int8.engine `
    --check-ort-cpu `
    --check-ort-cuda `
    --no-check-ort-tensorrt `
    --ort-run-dummy `
    --no-parity `
    --continue-on-error `
    --mlflow `
    --mlflow-tracking-uri "http://127.0.0.1:5000" `
    --mlflow-experiment-name "edge-perception-cpp" `
    --mlflow-run-name "yolo26m_bdd100k_eval_final_allconv_int8" `
    --mlflow-log-artifacts
```

Evaluation outputs include:

```text
evaluation_summary.json
artifact_reports.json
backend readiness checks
per-artifact validation folders
optional prediction JSON files
MLflow metrics and artifacts
```

---

## ONNX Runtime and TensorRT Notes

This project distinguishes between three different deployment routes:

```text
ONNX file + ONNX Runtime CPU Execution Provider
ONNX file + ONNX Runtime CUDA Execution Provider
native TensorRT .engine file
```

The ONNX Runtime TensorRT Execution Provider is not treated as a required backend for the final matrix because it was environment-sensitive on Windows and native TensorRT engines already cover the TensorRT deployment path.

Native TensorRT `.engine` files are usually specific to:

```text
GPU architecture
TensorRT version
CUDA version
operating system
driver/runtime environment
input shape/profile
```

Do not assume a `.engine` file exported on one machine will run unchanged on another machine.

---

## Parity Testing

Dedicated parity testing is planned for:

```text
tests/test_parity.py
```

The evaluation script also supports optional final-detection parity checks, but they are disabled in the final evaluation run.

Reason:

```text
Final detections are sensitive to small numeric differences.
Backend differences can change confidence ordering, NMS behavior, or top-k selection.
Strict parity can fail even when full mAP validation succeeds.
```

Parity is useful for debugging, not for the main quality matrix.

---

## Native C++ Inference Plan

The native C++ inference pipeline will use:

```text
C++17 or newer
OpenCV
ONNX Runtime C++
TensorRT runtime path
CMake
```

Core inference flow:

```text
load image / receive frame
↓
preprocess with OpenCV
↓
write into model input buffer
↓
run backend inference
↓
decode final detections
↓
apply confidence filtering
↓
apply NMS if needed
↓
scale boxes back to original image
↓
return detections
```

Planned components:

```text
ImageProcessor
InferenceEngine
Postprocess
Benchmark
```

---

## Preprocessing

OpenCV loads images as:

```text
HWC BGR uint8
```

The exported YOLO model expects:

```text
NCHW RGB float32 or float16 depending on artifact
```

Preprocessing must handle:

```text
resize or letterbox
BGR to RGB conversion
uint8 to float conversion
normalization to 0.0-1.0
HWC to NCHW conversion
batch dimension
letterbox metadata for box rescaling
```

The optimized C++ path should minimize unnecessary memory allocations and copies.

---

## Postprocessing

The exported model returns final detections:

```text
[1, 300, 6]
```

where each row is:

```text
[x1, y1, x2, y2, confidence, class_id]
```

Postprocessing should support:

```text
confidence filtering
class ID handling
box clipping
box scaling back to original image space
optional NMS if needed for future export variants
optional drawing
```

Drawing should be excluded from benchmark timing unless explicitly reported separately.

---

## Benchmarking Plan

C++ benchmarking should measure:

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
use the same input size
use the same confidence threshold
use the same IoU threshold
warm up before timing
separate engine build time from steady-state inference
do not include drawing unless reported separately
report mean, median, p95, p99
```

Only the C++ benchmark should make final claims about end-to-end deployment latency.

---

## Git / Artifact Policy

Commit source code, configs, and documentation.

Do not commit:

```text
datasets
model weights
ONNX files
TensorRT engines
MLflow databases
MLflow run directories
evaluation output folders
logs
temporary backup files
```

Generated local artifacts are reproducible from the scripts and should remain outside Git.

---

## Final Useful Files to Keep Locally

ONNX:

```text
models/onnx/yolo26m_bdd100k_fp32.onnx
models/onnx/yolo26m_bdd100k_fp16.onnx
models/onnx/yolo26m_bdd100k_allconv_int8.onnx
```

TensorRT:

```text
models/engine/yolo26m_bdd100k_fp32.engine
models/engine/yolo26m_bdd100k_fp16.engine
models/engine/yolo26m_bdd100k_int8.engine
```

Checkpoint:

```text
runs/train/yolo26m_bdd100k/weights/best.pt
```

These are local generated artifacts and should not be committed.

---

## Project Takeaway

This project demonstrates more than a standard YOLO training run.

It includes:

```text
large-scale dataset handling
detector training
MLflow experiment tracking
multi-format model export
ONNX Runtime backend validation
native TensorRT engine validation
INT8 quantization debugging
backend quality comparison
planned C++ deployment path
```

A key engineering result was diagnosing why naive ONNX INT8 failed, then replacing it with all-Conv ONNX INT8 QDQ that quantized all convolutional nodes while preserving a valid detection output.
