# edge-perception-cpp

Native C++ object detection inference pipeline using ONNX Runtime and OpenCV.

This project trains a YOLO26M detector on BDD100K driving-scene data, exports the trained model to ONNX, and runs inference through a native C++ runtime. The focus is the deployment path: preprocessing images with OpenCV, executing the model with ONNX Runtime, decoding detections, applying postprocessing, and benchmarking inference latency.

---

## Pipeline

```text
BDD100K Kaggle dataset
→ dataset download and validation
→ YOLO26M training/fine-tuning
→ model evaluation
→ ONNX export
→ PyTorch vs ONNX parity check
→ C++ ONNX Runtime inference
→ OpenCV preprocessing
→ detection postprocessing
→ latency benchmarking
→ optional INT8 quantization
```

---

## Project Status

```text
[done] data/download.py
[done] README.md
[pending] .gitignore
[pending] requirements.txt
[pending] data/dataset.py
[pending] training/train_detector.py
[pending] training/evaluate.py
[pending] training/export_onnx.py
[pending] tests/test_parity.py
[pending] C++ inference engine
[pending] benchmarking
[pending] optional INT8 quantization
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
│   ├── export_onnx.py
│   └── evaluate.py
├── src/
│   ├── main.cpp
│   ├── inference_engine.cpp
│   ├── image_processor.cpp
│   └── postprocess.cpp
├── include/
│   ├── inference_engine.hpp
│   └── image_processor.hpp
├── tests/
│   ├── test_inference.cpp
│   └── test_parity.py
├── CMakeLists.txt
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

The project uses a BDD100K dataset hosted on Kaggle.

The current download script is designed for a YOLO-format BDD100K dataset, meaning the downloaded dataset is expected to contain:

```text
images/
labels/
data.yaml
```

Default Kaggle dataset slug:

```text
a7madmostafa/bdd100k-yolo
```

After running the download script, the expected project-local dataset path is:

```text
data/raw/bdd100k/bdd100k/
```

Expected layout:

```text
data/raw/bdd100k/
├── kaggle_download/
├── bdd100k/
│   ├── images/
│   ├── labels/
│   └── data.yaml
└── download_manifest.json
```

Large dataset files are not committed to Git.

---

## Dataset Download

Dataset download is handled by:

```text
data/download.py
```

The script:

```text
downloads the Kaggle dataset
extracts the downloaded archive
creates a stable local dataset directory
finds the images directory
finds the labels directory
finds the dataset YAML
validates that the dataset is usable
writes download_manifest.json
```

Install the Kaggle CLI:

```bash
pip install kaggle
```

Configure the Kaggle API token before running the script.

Run the default download:

```bash
python data/download.py
```

Run a clean re-download:

```bash
python data/download.py --clean --force-download
```

Validate existing data without downloading again:

```bash
python data/download.py --validate-only
```

Use a different Kaggle dataset slug:

```bash
python data/download.py --kaggle-dataset <owner>/<dataset-name>
```

---

## Download Manifest

After dataset preparation, the script writes:

```text
data/raw/bdd100k/download_manifest.json
```

The manifest records:

```text
Kaggle dataset slug
raw directory
download directory
stable dataset directory
resolved images directory
resolved labels directory
resolved dataset YAML path
image count
label count
validation result
creation timestamp
```

---

## Training

Training will be handled by:

```text
training/train_detector.py
```

Planned model:

```text
YOLO26M
```

Expected training dataset input:

```text
data/raw/bdd100k/bdd100k/data.yaml
```

The training script will be responsible for:

```text
loading pretrained YOLO26M weights
loading the BDD100K dataset YAML
configuring training settings
fine-tuning the detector
saving checkpoints
saving metrics
saving sample predictions
```

---

## Evaluation

Evaluation will be handled by:

```text
training/evaluate.py
```

Planned outputs:

```text
mAP@50
mAP@75
precision
recall
per-class metrics
metrics JSON
sample prediction images
```

---

## ONNX Export

ONNX export will be handled by:

```text
training/export_onnx.py
```

Planned export target:

```text
FP32 ONNX
```

The export script should record:

```text
exported ONNX path
input tensor name
input tensor shape
output tensor names
output tensor shapes
ONNX Runtime smoke-test result
```

Optional later exports:

```text
FP16 ONNX
INT8 ONNX
```

---

## Parity Testing

PyTorch vs ONNX parity will be checked by:

```text
tests/test_parity.py
```

The test should compare model outputs between:

```text
YOLO/PyTorch
ONNX Runtime
```

The comparison may include:

```text
box coordinates
confidence scores
class IDs
number of detections
max absolute difference
mean absolute difference
```

---

## C++ Inference

The native inference pipeline will use:

```text
OpenCV
ONNX Runtime C++
C++17 or newer
```

Planned inference flow:

```text
load image with OpenCV
↓
resize or letterbox
↓
convert BGR HWC uint8 to RGB NCHW float32
↓
run ONNX Runtime inference
↓
decode model outputs
↓
apply confidence threshold
↓
apply NMS if required
↓
draw detections
↓
save output image
↓
report latency
```

---

## Image Preprocessing

OpenCV loads images as:

```text
HWC BGR uint8
```

YOLO-style ONNX models typically expect:

```text
NCHW RGB float32
```

The preprocessing implementation must handle:

```text
BGR to RGB conversion
resize or letterbox
uint8 to float32 conversion
normalization
HWC to NCHW layout conversion
batch dimension creation
```

The optimization target is a reusable preallocated input buffer for reducing repeated allocations during inference.

---

## Benchmarking

Benchmarking will compare:

```text
Python YOLO inference
Python ONNX Runtime inference
C++ ONNX Runtime inference
optional C++ ONNX Runtime INT8 inference
```

Measured components:

```text
preprocessing latency
inference latency
postprocessing latency
end-to-end latency
FPS
model size
memory usage
```

No benchmark numbers should be reported unless measured locally.

---

## Artifact Policy

The repository should track source code, configuration, and documentation.

The repository should not track large generated artifacts such as:

```text
data/raw/
data/processed/
runs/
mlruns/
models/
weights/
checkpoints/
*.pt
*.onnx
outputs/
benchmarks/
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
8. training/export_onnx.py
9. tests/test_parity.py
10. CMakeLists.txt
11. include/image_processor.hpp
12. src/image_processor.cpp
13. include/inference_engine.hpp
14. src/inference_engine.cpp
15. src/postprocess.cpp
16. src/main.cpp
17. benchmark mode
18. preprocessing optimization
19. optional INT8 quantization
20. tests/test_inference.cpp
21. Dockerfile
```

---

## Current Focus

Current completed files:

```text
data/download.py
README.md
```

Next files:

```text
.gitignore
requirements.txt
```
