# INT8 deployment and evaluation

The native and fused loaders accept raw TensorRT plans and Ultralytics' envelope
(`uint32` little-endian JSON length, JSON object, unchanged plan). Metadata is
validated and removed, not interpreted as permission to change runtime settings.
This does not make engines portable across TensorRT versions, GPUs or operating
systems. FP16 uses the same loader. Original model files are never modified.

## Run the workflow

From the repository root on a configured Windows workstation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_int8_pipeline.ps1
```

This is the **full, potentially long workflow**, not a smoke test. It prepares
the NVIDIA build, runs regression tests, selects calibration data, produces QDQ
ONNX, builds a target-specific engine, checks parity, evaluates the entire
prepared test split and measures both precisions. It does not train, use Docker,
install dependencies, commit, or upload anything. Keep other GPU workloads idle.
The [all-model benchmark](BENCHMARKING.md) is separate from this artifact-generation
workflow. Its final measurements cover the accepted canonical artifacts and all
other supported deployment paths.

Prerequisites: the repository `.venv`, trained checkpoint, existing FP32 ONNX and
FP16 engine, prepared BDD100K YAML/images/labels, NVIDIA driver, TensorRT **10.x**,
CUDA, OpenCV, ONNX Runtime and MSVC/Ninja/CMake. SDK discovery reuses the FP16
runner's local build hints or explicit `EDGE_*` environment overrides. Missing
files, incompatible graphs, build failures and failed execution stop visibly.
Paths, device, workspace budget and build directory are configurable parameters
in `tools/run_int8_pipeline.ps1`. The default build is `build/int8-deployment`;
previous FP16 executables and benchmark evidence are preserved.
All relative path arguments are resolved against the repository, even when the
command is launched from another directory. Absolute paths are preserved.
`-CheckPathsOnly` prints those resolved paths without creating outputs, building,
calibrating or evaluating; it is useful for checking a workstation configuration.

## Quantization policy and the original defect

The original ONNX INT8 export quantized the final `[1,300,6]` output containing
box coordinates, confidence and class IDs with one scale **2.8979830742** and
zero-point **−108**. Every confidence in `[0,1]` rounds to zero at that scale.
A real-image probe confirmed maximum confidence **0.0**. Its historical zero
mAP is not evidence of an invalid NMS threshold or broken INT8 engine loading.

The PTQ policy:

- Uses 1,024 training images by default, deterministic seed 42. A seeded
  candidate pool covers available classes and brightness bins. The selection,
  missing classes, hashes and labels are saved. This is a diversity proxy, not
  guaranteed coverage of every weather, camera or driving condition.
- Never falls back to validation/test calibration. Selected held-out content
  overlap is rejected. Quality uses the test split; parity/timing use validation.
- Matches deployment preprocessing: OpenCV decode, square 640×640 letterbox,
  padding 114, BGR→RGB, contiguous NCHW FP32 and division by 255. The previous
  calibration reader stretched images, which did not match deployment geometry.
- Quantizes identified backbone/neck **Conv** nodes with symmetric signed INT8
  activations (per tensor) and weights (per channel), MinMax calibration.
  The complete detection head, decode/selection and mixed-unit final output
  remain floating point; bias remains floating point. Unknown head naming fails
  rather than guessing exclusions. The older general exporter also limits QDQ
  to Conv, but does not provide this full head-exclusion/evaluation workflow.
- Builds TensorRT from explicit QDQ without a second implicit calibrator;
  unquantized regions may use FP16. External input/output remain FP32. Layer
  inspector JSON records the actual engine. This is **mixed precision**, not a
  claim that every operation or transfer uses INT8.

Symmetric QDQ is required by this TensorRT path. Legacy asymmetric ONNX models
are not silently rewritten or routed through ORT TensorRT EP. That C++ provider
policy remains unchanged. ONNX/Python evaluation may use mixed CPU/CUDA kernels;
it is not a native TensorRT performance result. ORT's optimized x86 integer
kernels can also introduce saturation (reproduced on an all-positive toy
fixture); the toy regression separately verifies unfused QDQ mathematics.
Full quality evaluation must still check the actual runtime path.

References: [ORT quantization and accuracy debugging](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html),
[TensorRT quantized types and explicit QDQ](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html).

## Validation and reporting

Every full run records:

- Same-engine native/fused/Python strict parity on 64 images; raw versus prefixed
  INT8 equivalence; FP16 regressions; INT8 versus FP16/PyTorch differences.
  Existing strict gates are unchanged. Empty detections are valid; missing
  records, invalid boxes/classes or nonfinite values are errors.
- Full prepared-test mAP50 and mAP50–95 for PyTorch, ONNX INT8, Python TRT
  INT8/FP16 and C++ native/fused INT8/FP16. Predictions use confidence 0.001,
  max 300 detections, square input and original-image coordinates. One common
  Ultralytics matching/AP implementation scores all these predictions.
- Sequential mean/p50/p95 and streaming depth 1/3 for both C++ backends and
  precisions: 128 identical decoded validation images, batch 1, 50 warmups,
  300 sequential or 1,000 streaming frames, three repetitions. Stage timers
  are off; graph capture is requested and capture failure is fatal. Cells run
  serially; order reverses between repetitions. Reports include raw samples,
  per-run statistics, ranges/CV and equal-depth INT8/FP16 comparisons.

Timing excludes decoding, initialization, drawing and disk I/O. Streaming frame
residence is not sequential latency. Native depth 3 uses independent workers,
contexts, streams and buffers; it can increase memory use and need not be faster.
The benchmark validates actual slot use/concurrent native worker activity.

Default acceptance requires all same-artifact parity gates and at most **2.0
percentage points** mAP50–95 loss for each C++ INT8 path versus its matching
FP16 path. `-MaxMapDrop` is an explicit configurable engineering budget, not a
lossless-quantization promise. Failed cross-precision strict comparisons remain
visible even if aggregate quality passes. Speed cannot override accuracy failure.

Outputs: `benchmarks/int8_pipeline_<timestamp>/pipeline/README.md`, `summary.json`,
`manifest.json`, per-stage logs, models, predictions and raw measurements.
Exit 0 means completed and accepted (or successful diagnostic smoke); exit 2
means completed with acceptance failures; exit 1 means interrupted/failed work.
Only full, identity-validated, accepted results can support deployment claims.

## Resume and reuse

Append `-Resume -ResultsDirectory '<the original evidence directory>'` to the
**same command/options**. Finished stages are hash-checked and reused. Prediction
jobs checkpoint complete 256-image chunks; interrupted chunks are retried in
new attempt directories. Calibration/engine building restart only their own
incomplete stage, not already completed stages. Failed/partial evidence is kept.
Never edit source, models, runtime or settings between starting and resuming;
changed identities require a new evidence directory.

`-ReuseInt8Engine` and `-ReuseInt8Onnx` explicitly reuse existing artifacts. An
inherited engine is reported as such, **not** as built/calibrated from the new
ONNX or calibration sample. `-Reexport` regenerates FP32 ONNX from the checkpoint;
otherwise the existing static FP32 export is reused and its identity recorded.

`-Smoke` reuses the existing INT8 engine and all-convolution ONNX, tests only four
parity/test images, and measures six frames per cell. It never runs full PTQ or
provides publishable quality/performance results. Current validation evidence and
remaining limitations are in [INT8 status](results/int8-status.md).

## Optional QAT boundary

Measured native/fused PTQ mAP50–95 is 33.2609 versus FP16 34.0120, within the
declared 2.0-point budget. This does not establish that QAT would improve it.
There is no implemented or executed YOLO26 QAT trainer in this project.
`-Mode qat -QatOnnx '<file>' -QatProvenance '<json>'` supports evaluation/deployment
of an independently trained, compatible QDQ artifact; it is not training.

The JSON must declare `method: "qat"`, matching `onnx_sha256` and
`base_checkpoint_sha256`, plus nonempty `training_command` and
`training_evidence`. These are provenance declarations, not proof of convergence.
The imported graph must satisfy the same symmetric QDQ, floating output and
fixed-shape checks, then pass the same parity/quality/performance workflow.
Do not relabel PTQ as QAT. If PTQ exceeds the accuracy budget, first inspect
per-class losses, calibration coverage and exclusions; only then select and
validate a YOLO26-compatible QAT training integration as a separate task.

## Activate accepted artifacts

After a full run reports `accepted: true` and `evidence_validated: true`, promote
its artifacts into the canonical inventory:

```powershell
python tools/promote_int8_artifacts.py --run "benchmarks/int8_pipeline_<run-id>/pipeline"
```

Replace `<run-id>` with the completed directory name. Promotion verifies hashes,
preserves previous canonical files/metadata under `models/archive/`, and writes
`models/int8_active.json`. It does not discard the original job outputs. An
existing different promotion record requires review rather than silent replacement.
The all-model runner checks this registry before measuring the active inventory.

The accepted preparation used 1,024 training images, 88 quantized Conv nodes and
24 excluded detection-head Conv nodes. [Final quality/performance](results/int8-status.md)
comes from the later matched all-model evaluation, not a mix of generation-run
and benchmark-run timings. Superseded zero-mAP exports are not active inputs.
