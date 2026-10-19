"""All-model coverage, provenance and launcher contracts; no real evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from tools import all_models as suite
from tools import all_models_worker as worker
from tools import promote_int8_artifacts as promotion
from tools.benchmark_release import file_record, measurement_plan, save_json


def fixture_inventory(tmp_path):
    models = tmp_path / 'models'
    for kind in ('onnx', 'engine'):
        (models / kind).mkdir(parents=True)
        for precision in ('fp32', 'fp16', 'int8'):
            (models / kind / f'yolo26m_bdd100k_{precision}.{kind}').write_bytes((kind + precision).encode())
    (models / 'onnx/yolo26m_bdd100k_allconv_int8.onnx').write_bytes(b'legacy')
    (models / 'onnx/yolo26m_bdd100k_convonly_fp32.onnx').write_bytes(b'onnxfp32')
    (models / 'archive').mkdir()
    (models / 'archive/old_int8.engine').write_bytes(b'not active')
    weights = tmp_path / 'best.pt'
    weights.write_bytes(b'checkpoint')
    save_json(models / 'int8_active.json', {'active_artifacts': [file_record(models / kind /
        f'yolo26m_bdd100k_int8.{kind}') for kind in ('onnx', 'engine')]})
    return models, weights


def test_inventory_covers_distinct_bytes_aliases_and_policy(tmp_path):
    artifacts = suite.inventory(*fixture_inventory(tmp_path))
    assert len(artifacts) == 8
    assert len(next(a for a in artifacts if a['label'] == 'onnx_fp32')['aliases']) == 1
    cells = suite.cells(artifacts)
    assert len(cells) == 18
    assert sum(c['supported'] for c in cells) == 16
    assert {c['label'] for c in cells if not c['supported']} == {'onnx_int8__ort-trt', 'onnx_allconv_int8__ort-trt'}
    assert not any('archive' in a['path'] for a in artifacts)


def test_inventory_rejects_changed_active_int8(tmp_path):
    models, weights = fixture_inventory(tmp_path)
    (models / 'onnx/yolo26m_bdd100k_int8.onnx').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed/missing'):
        suite.inventory(models, weights)


@pytest.mark.parametrize('precision', ['fp32', 'fp16', 'int8'])
def test_plan_includes_real_native_and_fused_depths(precision):
    for backend in ('native-trt', 'fused-trt'):
        plan = measurement_plan(backend, precision, True)
        assert ('latency_e2e_clean', 1) in plan
        assert ('throughput_e2e_clean', 1) in plan
        assert ('throughput_e2e_clean', 3) in plan
    assert all(depth == 1 for _, depth in measurement_plan('ort-cuda', precision, True))
    assert ('latency_backend_clean', 1) not in measurement_plan('fused-trt', precision, True)


def test_graph_ablation_is_explicit():
    assert ('throughput_e2e_graph_off', 3) in measurement_plan('fused-trt', 'fp16', True)
    assert not any('graph_off' in name for name, _ in measurement_plan('fused-trt', 'fp16', False))


def promotion_fixture(tmp_path):
    run = tmp_path / 'accepted/pipeline'
    run.mkdir(parents=True)
    stages = {}
    for name, suffix, key in [('quantize', 'onnx', 'onnx'), ('engine', 'engine', 'engine')]:
        path = run / ('new.' + suffix)
        path.write_bytes(('accepted' + suffix).encode())
        stages[name] = {'status': 'completed', 'outputs': [file_record(path)],
                        'result': {key: str(path), 'policy': {'head_excluded': True}}}
    save_json(run / 'manifest.json', {'status': 'completed', 'evidence_validated': True,
                                     'diagnostic_only': False, 'stages': stages})
    save_json(run / 'summary.json', {'accepted': True})
    return run


def test_promotion_archives_then_verifies_and_is_idempotent(tmp_path):
    run = promotion_fixture(tmp_path)
    models = tmp_path / 'models'
    (models / 'onnx').mkdir(parents=True)
    old = models / 'onnx/yolo26m_bdd100k_int8.onnx'
    old.write_bytes(b'original')
    first = promotion.promote(run, models)
    assert old.read_bytes() == b'acceptedonnx'
    assert Path(first['archived_files'][0]['path']).read_bytes() == b'original'
    assert promotion.promote(run, models) == first
    assert (run / 'new.onnx').read_bytes() == b'acceptedonnx'


def test_promotion_rejects_unaccepted_result(tmp_path):
    run = promotion_fixture(tmp_path)
    save_json(run / 'summary.json', {'accepted': False})
    with pytest.raises(ValueError, match='accepted full run'):
        promotion.promote(run, tmp_path / 'models')
    assert not (tmp_path / 'models').exists()


@pytest.mark.skipif(os.name != 'nt', reason='PowerShell path regression')
def test_launcher_is_independent_of_callers_directory(tmp_path):
    process = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        str(ROOT / 'tools/run_all_models.ps1'), '-CheckPathsOnly', '-ResultsDirectory',
        'benchmarks/all_path_check', '-RuntimeDirectories', 'third_party/runtime'],
        cwd=tmp_path, capture_output=True, text=True, check=True, timeout=30)
    paths = json.loads(process.stdout)
    assert Path(paths['python']) == ROOT / '.venv/Scripts/python.exe'
    assert Path(paths['models']) == ROOT / 'models'
    assert Path(paths['executable']) == ROOT / 'build/int8-deployment/bin/Release/edge_perception.exe'
    assert Path(paths['results']) == ROOT / 'benchmarks/all_path_check'
    assert paths['runtime_directories'] == [str(ROOT / 'third_party/runtime')]
    assert not (ROOT / 'benchmarks/all_path_check').exists()


@pytest.mark.skipif(os.name != 'nt', reason='Windows PowerShell 5.1 array regression')
@pytest.mark.parametrize('launcher', ['run_all_models.ps1', 'run_int8_pipeline.ps1'])
def test_runtime_json_array_is_not_nested_in_powershell(launcher, tmp_path):
    paths = ['E:/runtime with spaces', 'C:/cuda/bin']
    save_json(tmp_path / 'runtime-directories.json', paths)
    # Exercise the actual loader statement, without starting preparation/jobs.
    statement = next(line for line in (ROOT / 'tools' / launcher).read_text().splitlines()
                     if line.startswith('$runtime = '))
    command = "$prepare = '" + str(tmp_path).replace("'", "''") + "'; " + statement
    command += "; [ordered]@{count=$runtime.Count; path=($runtime -join ';')} | ConvertTo-Json"
    process = subprocess.run(['powershell', '-NoProfile', '-Command', command],
                             capture_output=True, text=True, check=True, timeout=30)
    value = json.loads(process.stdout)
    assert value == {'count': 2, 'path': ';'.join(paths)}


def test_performance_preserves_directory_interrupted_before_manifest(tmp_path, monkeypatch):
    old = tmp_path / 'performance'
    old.mkdir()
    (old / 'partial.log').write_text('preserve')
    output = tmp_path / 'job'
    output.mkdir()
    def execute(command, **kwargs):
        folder = Path(command[command.index('--output') + 1])
        assert folder == tmp_path / 'performance_attempt2'
        assert '--resume' not in command
        save_json(folder / 'manifest.json', {'status': 'completed', 'cells': [], 'failed_run_count': 0})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(worker.subprocess, 'run', execute)
    result = worker.performance({'directory': str(old), 'executable': 'app.exe', 'source': 'frames',
        'models': 'models', 'device': 0, 'smoke': True}, output)
    assert Path(result['manifest']).parent.name == 'performance_attempt2'
    assert (old / 'partial.log').read_text() == 'preserve'


def test_report_does_not_hide_failed_strict_parity_or_missing_quality(tmp_path):
    artifacts = suite.inventory(*fixture_inventory(tmp_path))
    manifest = {'status': 'running', 'diagnostic_only': True, 'artifacts': artifacts,
        'cells': suite.cells(artifacts), 'stages': {
            'compare_int8_pytorch': {'status': 'completed', 'result': {'matched_images': 0, 'images': 4, 'passed': False}},
            'quality_predict_python_pytorch': {'status': 'failed'}}}
    suite.render(tmp_path, manifest)
    report = (tmp_path / 'README.md').read_text(encoding='utf-8')
    assert '0/4 | FAIL' in report and 'FAILED' in report and 'PENDING' in report
    summary = suite.read(tmp_path / 'summary.json')
    assert summary['has_findings'] and not summary['evidence_validated']


def test_coordinator_requests_all_quality_paths_and_repetitions(tmp_path, monkeypatch):
    models, weights = fixture_inventory(tmp_path)
    artifacts = suite.inventory(models, weights)
    output = tmp_path / 'pipeline'
    output.mkdir()
    selection = tmp_path / 'selection.json'
    image = tmp_path / 'image.jpg'
    label = tmp_path / 'label.txt'
    image.write_bytes(b'image fixture')
    label.write_bytes(b'')
    row = {'key': '000000', **file_record(image), 'label': file_record(label)}
    save_json(selection, {key: [row] for key in ('parity', 'quality', 'benchmark')})
    runner = suite.AllModels.__new__(suite.AllModels)
    runner.output = output
    runner.args = SimpleNamespace(data=tmp_path / 'data.yaml', smoke=False, models_dir=models,
                                  weights=weights, executable=tmp_path / 'app.exe', device=0)
    runner.manifest = {'identity': {}, 'artifacts': artifacts, 'cells': suite.cells(artifacts), 'stages': {}}
    monkeypatch.setattr(suite, 'identity', lambda *args: {})
    requests = []
    def stage(name, action, **payload):
        requests.append((name, action, payload))
        return {'selection': str(selection)} if action == 'prepare' else {'chunks': name} if action == 'predict' else {}
    runner.stage = stage
    runner.attempt = stage
    runner.save = lambda: save_json(output / 'summary.json', {'has_findings': False})
    assert runner.run() == 0
    assert len([x for x in requests if x[1] == 'score']) == 24
    assert len([x for x in requests if x[1] == 'pytorch_timing']) == 3
    assert len([x for x in requests if x[1] == 'predict' and x[2]['split'] == 'parity']) == 24
    assert len([x for x in requests if x[1] == 'compare']) == 42
    assert next(x for x in requests if x[1] == 'prepare')[2]['evaluation_only'] is True
    assert not any(x[1] in ('export', 'quantize', 'engine') for x in requests)
