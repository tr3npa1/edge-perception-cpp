# Benchmarking and reproducibility

[Results](results/README.md) · [Interpretation](RESULTS_ANALYSIS.md) · [Setup](USAGE.md)

## Evaluation contract

The all-model suite evaluates the PyTorch checkpoint, FP32/FP16/INT8 ONNX and
TensorRT artifacts, and the separately labeled legacy all-convolution INT8 ONNX.
Byte-identical aliases are recorded rather than timed twice.

| Check | Coverage / protocol |
|---|---|
| Dataset quality | 8 Python artifacts + 16 supported C++ paths; 20,000 prepared test images each |
| Strict parity | 64 validation images, seed 42; same-artifact and PyTorch comparisons |
| Sequential latency | 128 frozen decoded validation images; 50 warmups, 300 measured frames, 3 repetitions |
| Streaming throughput | Same inputs; 50 warmups, 1,000 measured frames, 3 repetitions |
| Pipeline depths | Native TRT: 1/3; fused TRT: 1/2/3; ORT: 1 |
| Diagnostics | Separate 100-frame instrumented runs; fused FP16 graph-on/off comparison |
| Baseline | Separately timed synchronized Python/PyTorch API path |

All quality paths use square 640×640 input, batch 1, confidence 0.001 and at most
300 detections. One matching/AP implementation scores predictions in original
image coordinates. Parity uses confidence 0.25 and the
[unchanged strict tolerances](results/fp16-correctness.md#comparison-gates).

The prepared test split is not an official BDD100K leaderboard evaluation.
Calibration uses training images only; parity and timing use validation images.

## Timing boundaries

Sequential end-to-end latency measures decoded, in-memory BGR pixels through
preprocessing, inference and detection objects. Model loading, graph/session
creation, warmup, image decoding, annotation and disk output are excluded.

Streaming FPS is completed frames divided by measured wall time, including
draining. Streaming per-frame residence includes scheduling/backpressure; it
is neither sequential latency nor `1000 / FPS`. Backend-only timing is separate.
Stage diagnostics enable instrumentation and are not headline performance;
overlapping stage times must not be summed into wall latency.

Native depth 3 uses independent persistent workers, contexts, streams and
buffers. Fused slots use asynchronous submit/collect. Observed native worker
concurrency is checked; neither mechanism guarantees overlapping GPU kernels.
Fused graph capture is required when requested by the measurement protocol.

Summaries report medians of each per-run statistic, plus ranges and coefficient
of variation (CV). They are not pooled percentiles or confidence intervals.
Cell order reverses between repetitions. The published run resumed across
interruptions on an unlocked-clock workstation; small differences are not
claims of statistical significance.

## Full all-model run

Run commands from the repository root after [NVIDIA setup](USAGE.md#nvidia-build).
The launcher requires:

- Project `.venv/Scripts/python.exe`, CMake, Ninja, Git, MSVC and NVIDIA tools.
- Prepared dataset YAML/images/labels and the trained `best.pt`.
- FP32/FP16/INT8 ONNX and TensorRT files in `models/onnx/` and `models/engine/`.
- The legacy all-convolution INT8 ONNX for the published comparison coverage.
- Accepted canonical INT8 artifacts and their `models/int8_active.json` registry,
  created by [the promotion step](INT8_DEPLOYMENT.md#activate-accepted-artifacts).

A fresh clone is not a benchmark-ready installation: assets and SDKs are local.
The runner neither downloads artifacts nor calibrates/trains new ones.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_all_models.ps1
```

The launcher checks the inventory, configures/builds a separate
`build/int8-deployment` tree, runs regressions, then evaluates and measures.
Missing prerequisites fail visibly. A historical local build record can supply
SDK hints, but is not required when the `EDGE_*` variables in the setup guide
are configured. Override `-BuildConfiguration`, `-RuntimeDirectories`,
`-ModelsDirectory`, `-Weights`, `-Data`, `-BuildDirectory` or `-Device`
when needed. Relative paths resolve against the repository.

Useful diagnostics:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_all_models.ps1 -CheckPathsOnly
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_all_models.ps1 -Smoke
```

`-CheckPathsOnly` prints path resolution; it is not a dependency/execution test.
`-Smoke` exercises tiny samples and six timed frames; do not publish its values
as full quality/performance results. Full and smoke runs have distinct identities.

Full mode can take many hours, including CPU-only phases and ORT TensorRT cache
creation. **Wait for `ALL-MODEL SUITE FINISHED`, not zero GPU utilization.**

## Outputs and exit status

Each run owns a new `benchmarks/all_<run-id>/` directory:

```text
pipeline/manifest.json     Stage specifications, identities and checkpoints
pipeline/summary.json      Combined quality, parity and performance
pipeline/README.md         Human-readable run report
pipeline/jobs/             Predictions, comparisons, logs and attempts
performance/               Per-run measurements, aggregates and GPU telemetry
frames/                    Frozen timing inputs
prep_*/                    Build and regression-test evidence
```

Exit 0 means completed without reported findings. Exit 2 can mean a completed
suite with parity failures or unsupported combinations; it is not an all-pass
claim. An error stopping execution is distinct from a completed failed comparison.
Inspect `status`, `evidence_validated`, failed stages and individual results.

The QDQ INT8 ONNX / ORT TensorRT cells are rejected by project policy and receive
no fabricated timings. Native TensorRT INT8 uses its dedicated engine.

## Resume after interruption

Keep the original options and evidence directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_all_models.ps1 `
    -Resume -ResultsDirectory "benchmarks/all_<original-run-id>"
```

- Preparation checks rerun; completed stages and prediction chunks are reused
  only after their hashes are checked.
- Predictions checkpoint complete 256-image chunks. An interrupted chunk is
  retried; partial output is not accepted as complete.
- Finalized performance attempts, including recorded failures, are reused.
  An unfinished measurement is rerun in a new attempt directory.
- An OS-level lock prevents two coordinators from owning the same run.
- Source, artifact, runtime or configuration changes invalidate resume. Start
  new evidence instead of editing hashes or deleting failure records.

Power loss can corrupt even a recently written checkpoint. If JSON parsing or
hash validation fails, stop and inspect the exact affected stage. Preserve its
evidence before recovery; do not delete the whole run or bypass integrity checks.

## Focused workflows

- `tools/run_fp16_matrix.ps1 -FullMatrix`: fresh native/fused FP16 depths 1/3
  with separate sequential baselines. Without `-FullMatrix`, it runs native
  depth 3 and fused depth 1 plus latency baselines. This runner requires a new
  output directory; it does not support resume.
- `tools/run_int8_pipeline.ps1`: PTQ, engine generation, validation, full quality
  and FP16/INT8 performance. See [INT8_DEPLOYMENT.md](INT8_DEPLOYMENT.md).
- `tools/benchmark_release.py`: performance-only lower-level runner with
  `--resume`; it does not replace full quality/parity evaluation.

The older `run_full_benchmark.ps1`, `evaluate_release.py` and
`publish_results.py` serve the historical release layout. Use the all-model
entry point and exporter below for the current evidence format.

## Export a GitHub documentation bundle

The exporter performs no inference, build, upload or Git operation. It verifies
the completed run and recorded output hashes, retains failures, removes
workstation paths/identifiers, and writes compact JSON/CSV, plots and examples.
It needs Python 3.11+ and Matplotlib, plus the original local evidence.

```powershell
python docs/scripts/export_results.py `
    --source benchmarks/all_20261018_151027_308 `
    --output tmp/results-preview
```

Review the preview, then use `--output docs/results --replace` to replace only
the exporter's named generated files. The source run remains untouched.
The main README and technical guides are maintained separately.

Public checksums establish file identity, not independent reproduction.
Raw predictions, per-frame samples, weights, datasets and runtime caches remain
ignored and local. Published native Windows results do not validate Docker,
Linux GPU execution or a remote CI run.
