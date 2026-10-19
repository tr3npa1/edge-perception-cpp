# NVIDIA container deployment

[Overview](../README.md) · [Native setup](USAGE.md) · [Measured results](RESULTS_ANALYSIS.md)

The container packages the C++ command-line application, ONNX Runtime GPU,
TensorRT and CUDA. Models and inputs are mounted at runtime; they are not part
of the image. Inference writes annotated media and detection JSON to a host
output directory, then exits. This is a batch application, not an HTTP service.

## Published image

The versioned release is `tr3npa1/edge-perception-cpp:1.1.0-20261019` on
[Docker Hub](https://hub.docker.com/r/tr3npa1/edge-perception-cpp).
It targets Linux amd64 / CUDA architecture 75. To use it without compiling:

```text
docker pull tr3npa1/edge-perception-cpp:1.1.0-20261019
docker run --rm tr3npa1/edge-perception-cpp:1.1.0-20261019 --help
docker tag tr3npa1/edge-perception-cpp:1.1.0-20261019 edge-perception-cpp:local
```

The last command assigns the local alias used in the examples below. `latest`
is also published, but the versioned tag identifies this release. Models and
inputs are still required. The code is AGPL-3.0 with the project's limited
NVIDIA linking permission; the NVIDIA base and other dependencies retain their
own [license terms](../THIRD_PARTY_NOTICES.md). Corresponding project source and
notices are included in `/usr/share/edge-perception/source/`.

## Requirements and build

- Linux x86-64 containers and an NVIDIA GPU. Windows uses Docker Desktop's
  [WSL 2 GPU support](https://docs.docker.com/desktop/features/gpu/); Linux hosts
  need the NVIDIA Container Toolkit.
- The default CUDA architecture is **75** (RTX 2060). Set
  `--build-arg CUDA_ARCHITECTURES=<target>` for another supported GPU.
- The NVIDIA development base is large. Allow tens of GB for the image and
  build cache; Docker's disk location matters, not just the repository drive.
- Existing ONNX artifacts and your own input images. A Git clone alone does
  not include weights or a dataset.

From the repository root, in PowerShell or a Linux shell:

```text
docker build --build-arg BUILD_JOBS=2 -t edge-perception-cpp:local .
docker run --rm --network none edge-perception-cpp:local --help
docker run --rm --network none --gpus all --entrypoint nvidia-smi edge-perception-cpp:local
```

The default base is NVIDIA TensorRT `25.03-py3`, pinned by digest in the
Dockerfile: Ubuntu 24.04, CUDA 12.8 and TensorRT 10.9. The C++ ONNX Runtime GPU
SDK is 1.26.0 (CUDA 12), **not** the Windows CUDA-13 SDK used in the published
benchmark. OpenCV comes from Ubuntu packages. The base digest and top-level
Python test dependencies are pinned; apt packages are not a frozen snapshot,
so a future clean rebuild is not guaranteed to be byte-identical.

The build compiles native/fused backends with fast math disabled, uses an
isolated Python test environment, and requires all four CPU-safe CTest targets
to pass. A GPU is not needed during `docker build`. The final image runs as
non-root user `edge` (UID 10001); no privileged mode or Docker socket mount is
required. The allowlisted build context excludes models, data, credentials and
local benchmark outputs.

## Build a Linux engine

Do **not** mount a Windows `.engine` as a Linux deployment engine. Rebuild from
the portable ONNX artifact for the target GPU and TensorRT runtime. Keep Linux
engines separate from the Windows benchmark artifacts.

The following PowerShell commands use the canonical FP32 ONNX to build an
FP16-capable engine. Run the engine-building command once per target/model;
running it again overwrites that Linux engine, not the original Windows one.

```powershell
$root = (Get-Location).Path
$models = Join-Path $root 'models/onnx'
$engines = Join-Path $root 'models/container/linux-sm75-trt10.9'
New-Item -ItemType Directory -Force -Path $engines | Out-Null

docker run --rm --network none --gpus all `
    --mount "type=bind,source=$models,target=/models,readonly" `
    --mount "type=bind,source=$engines,target=/engines" `
    --entrypoint trtexec edge-perception-cpp:local `
    --onnx=/models/yolo26m_bdd100k_fp32.onnx `
    --saveEngine=/engines/yolo26m_bdd100k_fp16.engine `
    --fp16 --memPoolSize=workspace:1024 --skipInference
if ($LASTEXITCODE -ne 0) { throw 'Linux FP16 engine build failed.' }
```

`--skipInference` skips trtexec's benchmark, not the engine builder's tactic
selection. Engine construction can take several minutes. Keep its logs and
record model/engine hashes when producing evaluation evidence.

For the **accepted explicit-QDQ INT8 ONNX**, use:

```powershell
docker run --rm --network none --gpus all `
    --mount "type=bind,source=$models,target=/models,readonly" `
    --mount "type=bind,source=$engines,target=/engines" `
    --entrypoint trtexec edge-perception-cpp:local `
    --onnx=/models/yolo26m_bdd100k_int8.onnx `
    --saveEngine=/engines/yolo26m_bdd100k_int8.engine `
    --fp16 --memPoolSize=workspace:1024 --skipInference
if ($LASTEXITCODE -ne 0) { throw 'Linux INT8 engine build failed.' }
```

The INT8 scales are already embedded in Q/DQ nodes. `--fp16` allows the
unquantized regions to use FP16, matching the mixed-precision build policy;
it does not convert the quantized model back to a purely FP16 network. Do not
add implicit calibration or random INT8 scales. Prepare the accepted ONNX with
the [INT8 workflow](INT8_DEPLOYMENT.md) before this step; do not use the historical
all-convolution artifact as the canonical model.

## Run inference

Continue in the same PowerShell session, replacing the input directory with
an existing directory containing supported images:

```powershell
$inputs = (Resolve-Path 'C:\data\images').Path
$outputs = Join-Path $root 'outputs/container'
New-Item -ItemType Directory -Force -Path $outputs | Out-Null

docker run --rm --network none --gpus all `
    --mount "type=bind,source=$engines,target=/engines,readonly" `
    --mount "type=bind,source=$inputs,target=/inputs,readonly" `
    --mount "type=bind,source=$outputs,target=/outputs" `
    edge-perception-cpp:local `
    --mode infer --backend fused-trt --precision fp16 `
    --model /engines/yolo26m_bdd100k_fp16.engine `
    --source /inputs --pipeline-depth 3 --output-dir /outputs/fused-fp16
if ($LASTEXITCODE -ne 0) { throw 'Container inference failed.' }
```

For native TensorRT, change `--backend` to `native-trt`. For INT8, use the Linux
INT8 engine, `--precision int8`, and a separate output subdirectory. Add
`--json-only` to omit annotated images during directory inference.

CPU ONNX inference does not require `--gpus all`:

```powershell
docker run --rm --network none `
    --mount "type=bind,source=$models,target=/models,readonly" `
    --mount "type=bind,source=$inputs,target=/inputs,readonly" `
    --mount "type=bind,source=$outputs,target=/outputs" `
    edge-perception-cpp:local `
    --mode infer --backend ort-cpu --precision fp32 `
    --model /models/yolo26m_bdd100k_fp32.onnx `
    --source /inputs --output-dir /outputs/ort-cpu
if ($LASTEXITCODE -ne 0) { throw 'Container CPU inference failed.' }
```

Use `--gpus all` with `--backend ort-cuda` or `ort-trt` for explicit ONNX GPU
providers. An explicit backend must initialize successfully; it does not
silently switch to another backend. ORT TensorRT's first run builds a cache and
can take minutes. Add `--ort-cache-dir /outputs/ort-cache` to persist that cache
in the output mount; otherwise the container-local cache disappears on exit.
INT8 QDQ through ORT TensorRT remains intentionally rejected
by the project's routing policy; use native/fused TensorRT for that artifact.

On Linux, translate the mount source paths to absolute Linux paths and make the
output/engine directories writable by UID 10001, or run with your own UID/GID.
Do not make model or input mounts writable just to fix output permissions.

## Validation boundary

Container validation is tracked separately from the published Windows
benchmark. A successful build, smoke inference or parity regression does not
establish full-dataset Linux mAP or Linux latency/throughput. No registry push
is part of any command in this guide.

The local `edge-perception-cpp:local` image was verified on Docker Desktop
WSL 2 with an RTX 2060, using OpenCV 4.6.0 and the container stack above.
Exact image/model/engine hashes, detection counts, test groups and evidence
hashes are recorded in [docker-validation.json](docker-validation.json).

| Check | Verified outcome |
|---|---|
| Linux Release build and CPU-safe CTest | 4/4 targets passed |
| Non-root runtime | UID 10001; mounted output files saved successfully |
| ORT CPU, CUDA and TensorRT FP32 ONNX inference | 4/4 images processed by each explicit backend |
| Native and fused TRT FP16 inference | 4/4 images each; native/fused parity 4/4 |
| Native and fused TRT INT8 inference | 4/4 images each; native/fused parity 4/4 |
| FP16 and INT8 GPU correctness suites | 11/11 regression groups passed for each engine |
| QDQ INT8 ONNX through ORT TensorRT | Expected policy rejection, exit 1; no silent fallback |

The four inputs are the first four saved frames from the final Windows
benchmark, not selected by detection quality. Parity checks require equal
detection counts and classes, maximum coordinate error 0.002 pixels, maximum
confidence error 0.00001 and mean matched IoU at least 0.9999. Outputs were
also checked for finite values, valid classes and in-bounds boxes. Annotated
images and raw logs remain local under `outputs/docker-validation/`.

These are **same-engine native/fused** checks, not strict Windows/Linux or
cross-precision equivalence. FP16 and INT8 detection counts differ on this
sample. Full Linux mAP and performance, native/fused FP32 Linux engine inference,
and camera/video deployment are not validated by this check. The successful
pre-publication image was local only. The published release adds source/license
packaging and release labels; the original smoke evidence above is retained
without relabeling it as a new full benchmark. The published versioned tag is
separate from the older `1.1.0` image.

To run the existing GPU regression executable in the build stage:

```powershell
docker build --target builder --build-arg BUILD_JOBS=2 -t edge-perception-cpp:test-local .
docker run --rm --network none --gpus all `
    --mount "type=bind,source=$engines,target=/engines,readonly" `
    --mount "type=bind,source=$inputs,target=/inputs,readonly" `
    --entrypoint /workspace/build/bin/Release/edge_gpu_correctness_tests `
    edge-perception-cpp:test-local `
    --engine /engines/yolo26m_bdd100k_fp16.engine --image /inputs/frame.jpg
if ($LASTEXITCODE -ne 0) { throw 'Container GPU regressions failed.' }
```

Replace `frame.jpg` with an actual fixture filename. For INT8 use its engine
and append `--int8`. These regressions cover preprocessing, decoding,
determinism, graph/non-graph execution, slot lifetimes and native worker depths
1/3; their tiny scheduler timings are not publication benchmarks.
