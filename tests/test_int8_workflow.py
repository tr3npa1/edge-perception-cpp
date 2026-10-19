"""Focused INT8 contracts; toy calibration only, never the deployment model."""
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from training import int8_workflow as workflow
from training.export_model import YoloCalibrationDataReader, collect_calibration_images
from tools import int8_pipeline as pipeline, int8_worker as worker
from tools.benchmark_release import file_record, save_json


@pytest.mark.skipif(os.name != 'nt', reason='Windows PowerShell path regression')
def test_launcher_paths_do_not_depend_on_callers_directory(tmp_path):
    script = ROOT / 'tools/run_int8_pipeline.ps1'
    absolute_engine = tmp_path / 'engine with spaces.engine'
    process = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', str(script), '-CheckPathsOnly', '-ResultsDirectory', 'benchmarks/path-check-only',
        '-ReuseInt8Engine', str(absolute_engine), '-QatOnnx', 'models/qat.onnx',
        '-RuntimeDirectories', 'third_party/runtime'], cwd=tmp_path,
        capture_output=True, text=True, check=True, timeout=30)
    paths = json.loads(process.stdout)
    assert Path(paths['python']) == ROOT / '.venv/Scripts/python.exe'
    assert Path(paths['results']) == ROOT / 'benchmarks/path-check-only'
    assert Path(paths['build']) == ROOT / 'build/int8-deployment'
    assert Path(paths['weights']) == ROOT / 'runs/train/yolo26m_bdd100k/weights/best.pt'
    assert Path(paths['data']) == ROOT / 'data/processed/bdd100k_yolo/bdd100k.yaml'
    assert Path(paths['fp32_onnx']) == ROOT / 'models/onnx/yolo26m_bdd100k_fp32.onnx'
    assert Path(paths['fp16_engine']) == ROOT / 'models/engine/yolo26m_bdd100k_fp16.engine'
    assert Path(paths['build_configuration']) == ROOT / 'benchmarks/release_20261011/validation/nvidia-build.json'
    assert Path(paths['reuse_engine']) == absolute_engine
    assert Path(paths['qat_onnx']) == ROOT / 'models/qat.onnx'
    assert [Path(p) for p in paths['runtime_directories']] == [ROOT / 'third_party/runtime']
    assert not (tmp_path / 'benchmarks').exists()
    assert not (ROOT / 'benchmarks/path-check-only').exists()


def toy_model():
    nodes = [
        helper.make_node('Conv', ['images', 'body_w'], ['features'], name='/model.0/Conv', strides=[640, 640]),
        helper.make_node('Conv', ['features', 'head_w'], ['head'], name='/model.23/Conv'),
        helper.make_node('Reshape', ['head', 'shape'], ['rows'], name='/model.23/Reshape'),
        helper.make_node('Concat', ['rows', 'rows'], ['output0'], name='/model.23/Concat', axis=1)]
    graph = helper.make_graph(nodes, 'tiny_contract_fixture',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info('output0', TensorProto.FLOAT, [1, 300, 6])], [
            numpy_helper.from_array(np.full((2, 3, 1, 1), .1, np.float32), 'body_w'),
            numpy_helper.from_array(np.full((900, 2, 1, 1), .01, np.float32), 'head_w'),
            numpy_helper.from_array(np.array([1, 150, 6], np.int64), 'shape')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
    model.ir_version = 10
    return model


def image_fixture(tmp_path):
    path = tmp_path / 'image.png'
    image = np.full((32, 64, 3), (20, 80, 200), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def test_calibration_square_letterbox_rgb_layout_and_normalization(tmp_path):
    reader = YoloCalibrationDataReader([image_fixture(tmp_path)], 'images', 640)
    tensor = reader.get_next()['images']
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32 and tensor.flags.c_contiguous
    np.testing.assert_array_equal(tensor[0, :, 0, 0], np.full(3, np.float32(114) / 255))
    np.testing.assert_array_equal(tensor[0, :, 320, 320], np.array([200, 80, 20], np.float32) / 255)
    assert reader.get_next() is None


def test_unreadable_calibration_fails(tmp_path):
    with pytest.raises(ValueError, match='Cannot decode'):
        YoloCalibrationDataReader([tmp_path / 'missing.png'], 'images', 640).get_next()


def test_calibration_never_falls_back_to_validation(tmp_path):
    data = tmp_path / 'data.yaml'
    data.write_text(f'path: {tmp_path.as_posix()}\nval: .\nnames: [car]\n')
    with pytest.raises(ValueError, match='train split'):
        collect_calibration_images(data_yaml=data, fraction=1, max_images=1)


def test_policy_excludes_complete_detection_head():
    policy = workflow.quantization_policy(toy_model())
    assert policy['nodes_to_quantize'] == ['/model.0/Conv']
    assert policy['nodes_to_exclude'] == ['/model.23/Conv']


def test_unknown_detection_head_is_rejected():
    model = toy_model()
    model.graph.node[-1].name = 'unknown_output'
    with pytest.raises(ValueError, match='Unknown YOLO'):
        workflow.quantization_policy(model)


def test_real_tiny_ptq_symmetric_qdq_and_cpu_execution(tmp_path):
    import onnxruntime as ort
    source, target = tmp_path / 'float.onnx', tmp_path / 'int8.onnx'
    onnx.save(toy_model(), source)
    picture = image_fixture(tmp_path)
    result = workflow.quantize(source, target, [picture])
    assert result['graph']['nonzero_zero_points'] == 0
    assert result['graph']['outputs'][0]['producer'] == 'Concat'
    tensor = YoloCalibrationDataReader([picture], 'images', 640).get_next()
    # Evaluate QDQ graph semantics, not ORT's x86 U8S8 fused kernel: this
    # deliberately all-positive toy saturates that kernel's 16-bit accumulator.
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    actual = ort.InferenceSession(str(target), options, providers=['CPUExecutionProvider']).run(None, tensor)[0]
    expected = ort.InferenceSession(str(source), providers=['CPUExecutionProvider']).run(None, tensor)[0]
    assert actual.shape == (1, 300, 6) and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=.02, atol=.0001)


def test_mixed_output_scale_erases_all_probability_values():
    confidence = np.linspace(0, 1, 1001)
    scale, zero = 2.8979830741882324, -108
    decoded = (np.round(confidence / scale + zero) - zero) * scale
    assert np.count_nonzero(decoded) == 0


@pytest.mark.parametrize('failure', ['output', 'asymmetric', 'scales', 'no_qdq'])
def test_unsafe_qdq_rejected(monkeypatch, tmp_path, failure):
    from tools import audit_int8
    info = {'operators': {'QuantizeLinear': 1, 'DequantizeLinear': 1},
            'invalid_scales': [], 'nonzero_zero_points': 0, 'outputs': [{'producer': 'Concat'}]}
    if failure == 'output': info['outputs'][0]['producer'] = 'DequantizeLinear'
    if failure == 'asymmetric': info['nonzero_zero_points'] = 1
    if failure == 'scales': info['invalid_scales'] = ['bad']
    if failure == 'no_qdq': info['operators'] = {}
    monkeypatch.setattr(audit_int8, 'inspect', lambda _: info)
    with pytest.raises(ValueError):
        workflow.validate_qdq(tmp_path / 'unused.onnx')


def test_raw_and_prefixed_engine_payload(tmp_path):
    path = tmp_path / 'fixture.engine'
    raw = b'ptrt\x00\x01\x02'
    path.write_bytes(raw)
    assert workflow.engine_bytes(path) == (raw, None)
    metadata = b'{"names":{"0":"car"},"args":{"int8":true}}'
    path.write_bytes(len(metadata).to_bytes(4, 'little') + metadata + raw)
    assert workflow.engine_bytes(path) == (raw, json.loads(metadata))


@pytest.mark.parametrize('content', [b'', b'\xff\xff\xff\xff{}plan', b'\x02\0\0\0{}', b'\x03\0\0\0{!}plan'])
def test_malformed_engine_rejected(tmp_path, content):
    path = tmp_path / 'bad.engine'
    path.write_bytes(content)
    with pytest.raises(ValueError):
        workflow.engine_bytes(path)


@pytest.mark.parametrize('change', ['delete', 'edit'])
def test_completed_evidence_is_verified_on_resume(tmp_path, change):
    path = tmp_path / 'evidence.json'
    path.write_text('{}')
    records = [file_record(path)]
    pipeline.verify_outputs(records)
    if change == 'delete': path.unlink()
    else: path.write_text('{"modified":true}')
    with pytest.raises(ValueError, match='evidence changed/missing'):
        pipeline.verify_outputs(records)


@pytest.mark.parametrize('field,value', [('class_id', 99), ('confidence', float('nan')),
    ('confidence', 1.1), ('xyxy', [2, 0, 1, 4]), ('xyxy', [0, 0, float('inf'), 4])])
def test_invalid_detection_is_not_silently_scored(field, value):
    detection = {'class_id': 0, 'confidence': .5, 'xyxy': [0, 0, 2, 4]}
    detection[field] = value
    with pytest.raises(ValueError, match='Invalid detection'):
        worker.checked_predictions([detection], {0: 'car'})


def test_empty_detections_are_valid():
    assert worker.checked_predictions([], {0: 'car'}) == []


def test_smoke_or_partial_report_cannot_claim_acceptance(tmp_path):
    manifest = {'status': 'running', 'diagnostic_only': True, 'stages': {},
                'identity': {'settings': {'max_map_drop': 2}}, 'evidence_validated': False}
    pipeline.render(tmp_path, manifest)
    result = pipeline.read(tmp_path / 'summary.json')
    assert not result['accepted'] and not result['evidence_validated']
    report = (tmp_path / 'README.md').read_text(encoding='utf-8')
    assert 'SMOKE ONLY' in report and 'PENDING' in report


def test_prediction_resume_reuses_only_hash_verified_complete_chunks(tmp_path):
    picture = image_fixture(tmp_path)
    label = tmp_path / 'label.txt'
    label.write_text('0 .5 .5 .2 .2\n')
    selection = tmp_path / 'selection.json'
    save_json(selection, {'names': {0: 'car'}, 'parity': [
        {'key': '000000', **file_record(picture), 'label': file_record(label)}]})
    predictions = tmp_path / 'predictions.json'
    save_json(predictions, {'000000': []})
    save_json(tmp_path / 'chunks.json', {'0': file_record(predictions)})
    job = {'selection': str(selection), 'split': 'parity', 'backend': 'native-trt',
           'confidence': .25, 'label': 'test', 'executable': 'must-not-run'}
    assert worker.predict(job, tmp_path)['images'] == 1
    predictions.write_text('{}')
    with pytest.raises(ValueError, match='Input changed'):
        worker.predict(job, tmp_path)


def test_benchmark_writer_reader_directory_contract(tmp_path, monkeypatch):
    from tools import benchmark_release
    picture = image_fixture(tmp_path)
    selection = tmp_path / 'selection.json'
    save_json(selection, {'benchmark': [{'key': str(i), **file_record(picture)} for i in range(3)]})
    model = tmp_path / 'model.engine'
    def execute(command, folder, root, timeout):
        report = {'schema_version': 1, 'duration_unit': 'milliseconds',
            'status': 'completed', 'successful': True,
            'config': {'retain_raw_samples': True, 'failure_policy': 'stop_on_first_failure',
                'measured_iterations': 6, 'warmup_iterations': 3, 'mode': 'synchronous_latency',
                'timing_scope': 'end_to_end', 'pipeline_depth': 1, 'collect_stage_timings': False},
            'counts': {'warmup': {'completed': 3, 'failed': 0}, 'measurement': {'completed': 6, 'failed': 0}},
            'effective_pipeline_depth': 1, 'execution_policy': 'synchronous',
            'samples': [{'iteration_index': i, 'source_frame_index': i % 3} for i in range(6)],
            'primary_latency': {'sample_count': 6, 'mean_ms': 2., 'median_ms': 2., 'p95_ms': 3.},
            'throughput': {'completed_frames': 6, 'frames_per_second': 500.},
            'target': {'type': 'inference_engine', 'engine_config': {'artifact_path': str(model),
                'gpu': {'device_id': 0, 'enable_cuda_graph': True}},
                'runtime_info': {'backend': 'native_tensorrt', 'artifact_precision': 'int8',
                    'model': {'input_shape': [1, 3, 640, 640]}},
                'preprocess_config': {}, 'postprocess_config': {}}}
        # Respect the CLI's selected destination, so a disagreement with the
        # existing shared reader fails here instead of after a long evaluation.
        destination = Path(command[command.index('--output-dir') + 1])
        save_json(destination / 'case/benchmark.json', report)
        return {'exit_code': 0, 'timed_out': False}
    monkeypatch.setattr(benchmark_release, 'execute', execute)
    output = tmp_path / 'work'
    output.mkdir()
    actual = worker.benchmark({'selection': str(selection), 'smoke': True, 'mode': 'latency',
        'backend': 'native-trt', 'precision': 'int8', 'depth': 1, 'device': 0,
        'executable': 'unused', 'model': str(model)}, output)
    assert actual['fps'] == 500 and actual['p95_ms'] == 3
    assert Path(actual['raw_report']).is_file()
