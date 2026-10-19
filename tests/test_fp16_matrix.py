"""Fast runner contracts; no models, GPU execution or benchmark suite."""
from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import benchmark_fp16_matrix as matrix


def report(depth=3):
    return {"schema_version": 1, "duration_unit": "milliseconds",
            "config": {"retain_raw_samples": True, "failure_policy": "stop_on_first_failure"},
            "effective_pipeline_depth": depth, "execution_policy": "native_per_slot_workers",
            "maximum_active_workers": depth,
            "samples": [{"iteration_index": i, "pipeline_slot_index": i % depth, "source_frame_index": i % 3} for i in range(6)],
            "primary_latency": {"mean_ms": 2.0, "median_ms": 1.9, "p95_ms": 3.0},
            "throughput": {"frames_per_second": 300.0}}


def test_default_only_missing_streaming_cells():
    assert matrix.cells(False) == [("native-trt", "latency", 1), ("fused-trt", "latency", 1),
                                   ("native-trt", "throughput", 3), ("fused-trt", "throughput", 1)]
    assert len(matrix.cells(True)) == 6


def test_native_depth_three_accepts_real_workers():
    assert matrix.validate_report(report(), "native-trt", "throughput", 3, 3, 6) == []


def test_silent_serialization_rejected():
    errors = matrix.validate_report(report(1), "native-trt", "throughput", 3, 3, 6)
    assert any("silent serialization" in message for message in errors)
    assert any("concurrent active workers" in message for message in errors)


def test_missing_slot_and_wrong_inputs_rejected():
    value = report()
    value["samples"] = [{"pipeline_slot_index": 0, "source_frame_index": 0}] * 6
    errors = matrix.validate_report(value, "native-trt", "throughput", 3, 3, 6)
    assert any("slot" in message for message in errors)
    assert any("input sequence" in message for message in errors)


def test_variability_is_across_runs_not_pooled_samples():
    value = matrix.variability([1.0, 2.0, 3.0])
    assert value["median"] == 2.0
    assert value["sample_stddev"] == 1.0
    assert value["cv_percent"] == 50.0
    assert matrix.variability([1.0])["sample_stddev"] is None


def test_unexecuted_and_partial_cells_stay_pending():
    manifest = {"cells": matrix.cells(False), "protocol": {"repetitions": 3}, "runs": [
        {"backend": "native-trt", "mode": "latency", "depth": 1, "status": "passed",
         "metrics": {"mean_ms": 1.0, "median_ms": 1.0, "p95_ms": 1.1, "fps": 100.0}}]}
    rows = matrix.summarize(manifest)
    assert len(rows) == 6
    assert all(row["status"] == "PENDING" for row in rows)
    assert sum(row["requested"] for row in rows) == 4


def test_bad_numeric_metrics_rejected():
    value = report()
    value["primary_latency"]["p95_ms"] = float("nan")
    value["throughput"]["frames_per_second"] = 0
    assert len(matrix.validate_report(value, "native-trt", "throughput", 3, 3, 6)) == 2


def test_raw_report_wrong_config_and_false_completion(tmp_path):
    folder = tmp_path / 'app_output'
    folder.mkdir()
    (folder / 'benchmark.json').write_text(json.dumps({"successful": False, "status": "completed", "config": {"pipeline_depth": 1}}))
    _, errors = matrix.read_benchmark(tmp_path, 6, 3, 'throughput', 'end-to-end', 3, False)
    assert any('not successful' in error for error in errors)
    assert any('pipeline_depth' in error for error in errors)
    assert any('sample count' in error for error in errors)


def test_changed_source_and_binary_are_rejected(tmp_path, monkeypatch):
    binary = tmp_path / 'app.exe'
    binary.write_bytes(b'first')
    item = matrix.file_record(binary)
    manifest = {'executable': item, 'engine': item, 'source_snapshot': {'sha256': 'old'}}
    monkeypatch.setattr(matrix, 'source_snapshot', lambda root: {'sha256': 'new'})
    with pytest.raises(ValueError, match='Source changed'):
        matrix.verify_evidence(manifest, tmp_path)
    binary.write_bytes(b'changed')
    with pytest.raises(ValueError, match='executable changed'):
        matrix.verify_evidence(manifest, tmp_path)


def test_failed_provenance_does_not_publish_passed_numbers():
    manifest = {'status': 'failed', 'cells': matrix.cells(False), 'protocol': {'repetitions': 1}, 'runs': [
        {'backend': 'native-trt', 'mode': 'latency', 'depth': 1, 'status': 'passed',
         'metrics': {'mean_ms': 1.0, 'median_ms': 1.0, 'p95_ms': 1.0, 'fps': 1000.0}}]}
    assert matrix.summarize(manifest)[0]['status'] == 'UNVERIFIED'


def test_frame_order_change_is_rejected():
    value = report()
    value['samples'][0]['source_frame_index'] = 1
    assert any('input sequence' in message for message in matrix.validate_report(value, 'native-trt', 'throughput', 3, 3, 6))


def test_duplicate_samples_schema_and_policy_are_rejected():
    value = report()
    value['samples'][3]['iteration_index'] = 0
    value['schema_version'] = 999
    value['config']['failure_policy'] = 'continue_after_failure'
    value['execution_policy'] = 'synchronous'
    errors = matrix.validate_report(value, 'native-trt', 'throughput', 3, 3, 6)
    assert any('duplicate' in error for error in errors)
    assert any('schema' in error for error in errors)
    assert any('stop-on-first-failure' in error for error in errors)
    assert any('execution policy' in error for error in errors)


def test_diagnostic_flag_survives_machine_readable_report(tmp_path):
    manifest = {'status': 'completed', 'diagnostic_only': True, 'evidence_validated': True,
                'cells': matrix.cells(False), 'protocol': {'repetitions': 1}, 'runs': []}
    matrix.write_reports(tmp_path, manifest)
    assert json.loads((tmp_path / 'summary.json').read_text())['diagnostic_only'] is True
    assert 'DIAGNOSTIC ONLY' in (tmp_path / 'README.md').read_text(encoding='utf-8')


def test_wrong_fused_backend_or_cpu_postprocess_rejected():
    for field, value in [('type', 'inference_engine'), ('enable_gpu_postprocess', False),
                         ('require_cuda_graph', False)]:
        target = {'type': 'native_tensorrt_pipeline',
                  'pipeline_config': {'enable_gpu_postprocess': True, 'require_cuda_graph': True},
                  'runtime_info': {'gpu_postprocess_enabled': True}}
        (target if field == 'type' else target['pipeline_config'])[field] = value
        with pytest.raises(ValueError, match='Fused GPU'):
            matrix.semantics({'target': target}, 'fused-trt')
