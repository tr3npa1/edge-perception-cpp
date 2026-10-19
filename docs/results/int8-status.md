# INT8 deployment results

[Workflow](../INT8_DEPLOYMENT.md) · [All results](README.md)

Canonical mixed-precision PTQ artifacts have completed calibration, engine
generation, C++ deployment, strict same-engine checks and full-test evaluation.
They are distinct from the legacy all-convolution comparison and the archived
original zero-mAP export. No QAT training was performed.

## Quality and performance

All quality rows below use 20,000 prepared test images. Performance is the
matched final all-model run, not timings from the earlier artifact-generation run.

| Path | mAP50 | mAP50–95 | Sequential mean | p95 | Depth-3 FPS |
|---|---:|---:|---:|---:|---:|
| Native FP16 | 60.0445 | 34.0120 | 5.271 ms | 5.339 ms | 254.615 |
| Fused FP16 | 60.0445 | 34.0120 | 5.233 ms | 5.407 ms | 245.906 |
| Native INT8 | 58.9273 | 33.2609 | 5.185 ms | 5.246 ms | 278.790 |
| Fused INT8 | 58.9273 | 33.2609 | 4.929 ms | 4.973 ms | 278.814 |

INT8 reduces mAP50–95 by 0.7511 points versus the matching FP16 pipeline,
within the workflow's declared 2.0-point budget. This is an explicit engineering
tolerance, not lossless quantization. Fused INT8 has 13.38% higher depth-3
throughput than fused FP16; native/fused INT8 throughput is effectively tied.

ONNX INT8 is a different deployment result: ORT CUDA measures 32.9093 mAP50–95
and 42.921 ms sequential mean; CPU measures 32.9284 and 432.156 ms. Neither is a
speed improvement over its FP16/FP32 counterparts on this workstation.

## Correctness

Native/fused/Python TensorRT INT8 and direct native/fused comparisons pass
64/64 with the same engine. Strict TensorRT INT8 versus PyTorch passes 1/64.
Both are retained outcomes: same-engine deployment correctness does not imply
cross-precision numerical identity.

Canonical ONNX INT8 passes 64/64 against Python on ORT CUDA, but only 1/64 on
ORT CPU. The full CPU mAP is separately measured; that does not waive its
strict parity failure. INT8 ONNX via ORT TensorRT is rejected by routing policy,
while the native/fused INT8 engine paths execute successfully.

## Implemented corrections

- Both C++ loaders validate and strip an Ultralytics JSON metadata envelope
  before TensorRT deserialization; raw plans remain supported.
- The original final-output quantizer used scale 2.8979830742 with zero-point
  −108, rounding 0–1 confidence scores to zero. The active policy leaves the
  detection head and mixed-unit output floating point.
- Calibration uses deployment-matched square letterboxing rather than stretching,
  deterministic train-only selection and explicit symmetric QDQ.
- Accepted preparation used 1,024 training images, 88 quantized Conv nodes and
  24 excluded detection-head Conv nodes. Unquantized engine regions may use FP16;
  external bindings remain FP32.

Artifact identity and active deployment hashes are published in
[provenance.json](provenance.json); the local registry is `models/int8_active.json`.
Original evidence and superseded bytes remain local, outside active discovery.

## Limits

Calibration diversity is a proxy, not guaranteed coverage of every driving
condition. GPU architecture, TensorRT tactics and provider kernels can change
quality or speed. Imported QAT artifacts can be evaluated with provenance,
but there is no implemented YOLO26 QAT trainer or measured QAT improvement here.
The native Windows evidence does not validate Linux container deployment.
