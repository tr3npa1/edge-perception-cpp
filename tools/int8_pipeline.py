"""Resumable INT8 job coordinator. Full work runs only when explicitly invoked."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_release import OutputLock, build_snapshot, file_record, hardware_snapshot, save_json, sha256, utc_now
from tools.benchmark_fp16_matrix import source_snapshot, variability


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def fingerprint(args):
    code = source_snapshot(ROOT)
    extra = [p for folder in ('training', 'tools', 'tests') for p in (ROOT / folder).glob('*.py')]
    extra.append(ROOT / 'tools/run_int8_pipeline.ps1')
    extra.extend(p for p in (ROOT / 'tests').glob('*') if p.suffix in ('.cpp', '.hpp'))
    return {'code': code, 'python_code': {str(p.relative_to(ROOT)): sha256(p) for p in sorted(extra)},
        'inputs': {key: file_record(Path(value)) for key, value in vars(args).items()
                   if key in ('executable', 'weights', 'data', 'fp32_onnx', 'fp16_engine',
                              'reuse_int8_engine', 'reuse_int8_onnx', 'qat_onnx', 'qat_provenance') and value},
        'settings': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                     if key not in ('resume', 'output')},
        'build': build_snapshot(args.executable),
        'gpu_test': file_record(args.executable.with_name('edge_gpu_correctness_tests.exe')),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('onnx', 'onnxruntime-gpu', 'torch', 'ultralytics', 'numpy', 'tensorrt_cu12')}}


def verify_outputs(records):
    for record in records:
        if not Path(record['path']).is_file() or sha256(Path(record['path'])) != record['sha256']:
            raise ValueError(f"Completed evidence changed/missing: {record['path']}")


class Pipeline:
    def __init__(self, args):
        self.args, self.output = args, args.output.resolve()
        manifest_path = self.output / 'manifest.json'
        identity = fingerprint(args)
        if manifest_path.exists():
            if not args.resume:
                raise ValueError('Existing run requires --resume; never overwrite evidence')
            self.manifest = read(manifest_path)
            if self.manifest['identity'] != identity:
                raise ValueError('Resume rejected: source, artifacts, runtime or configuration changed')
        else:
            if args.resume:
                raise ValueError('No manifest to resume')
            self.manifest = {'schema_version': 1, 'created_utc': utc_now(), 'identity': identity,
                             'hardware': hardware_snapshot(ROOT), 'stages': {}, 'diagnostic_only': args.smoke}
        self.manifest.update(status='running', error=None, evidence_validated=False)
        self.save()

    def save(self):
        self.manifest['updated_utc'] = utc_now()
        save_json(self.output / 'manifest.json', self.manifest)
        render(self.output, self.manifest)

    def stage(self, name, action, **payload):
        stages = self.manifest['stages']
        spec = {'action': action, **payload}
        previous = stages.get(name)
        if previous and previous['spec'] != spec:
            raise ValueError(f'Changed stage specification: {name}')
        if previous and previous['status'] == 'completed':
            verify_outputs(previous['outputs'])
            print(f'REUSE {name}', flush=True)
            return previous['result']
        stage = previous or {'spec': spec, 'attempts': []}
        stages[name] = stage
        # Keep native C++ report paths below Windows MAX_PATH. Full stage names
        # remain in the manifest and job, not repeated in every nested path.
        job_id = hashlib.sha256(name.encode()).hexdigest()[:12]
        attempt = self.output / 'jobs' / job_id / f'a{len(stage["attempts"]) + 1}'
        attempt.mkdir(parents=True, exist_ok=False)
        # Only prediction jobs resume internal, hash-checked complete chunks.
        work = attempt.parent / 'predictions' if action == 'predict' else attempt / 'w'
        job = {**spec, 'output': str(work)}
        job_path = attempt / 'job.json'
        save_json(job_path, job)
        record = {'started_utc': utc_now(), 'directory': str(attempt)}
        stage['attempts'].append(record)
        stage['status'] = 'running'
        self.save()
        print(f'START {name}; log: {attempt / "console.log"}', flush=True)
        with (attempt / 'console.log').open('w', encoding='utf-8') as log:
            process = subprocess.run([sys.executable, str(getattr(self, 'worker', ROOT / 'tools/int8_worker.py')), '--job', str(job_path)],
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record.update(exit_code=process.returncode, finished_utc=utc_now())
        if process.returncode:
            stage['status'] = 'failed'
            self.save()
            raise RuntimeError(f'{name} failed ({process.returncode}); see {attempt / "console.log"}')
        result = read(work / 'result.json')
        outputs = [file_record(p) for p in work.rglob('*') if p.is_file()
                   and p.suffix in ('.json', '.jsonl', '.onnx', '.engine')]
        outputs.extend(result.get('external_outputs', []))
        stage.update(status='completed', result=result, outputs=outputs)
        self.save()
        return result

    def run(self):
        a = self.args
        prepared = self.stage('selection', 'prepare', data=str(a.data), seed=a.seed, smoke=a.smoke,
                              calibration_images=a.calibration_images)
        selection = prepared['selection']
        base = self.stage('export', 'export', weights=str(a.weights), fp32_onnx=str(a.fp32_onnx),
                          reexport=a.reexport, device=a.device)
        quantized = self.stage('quantize', 'quantize', onnx=base['onnx'], selection=selection,
            reuse_onnx=str(a.reuse_int8_onnx) if a.reuse_int8_onnx else None,
            qat_onnx=str(a.qat_onnx) if a.qat_onnx else None,
            qat_provenance=str(a.qat_provenance) if a.qat_provenance else None, weights=str(a.weights))
        built = self.stage('engine', 'engine', onnx=quantized['onnx'], selection=selection,
            reuse_engine=str(a.reuse_int8_engine) if a.reuse_int8_engine else None,
            workspace_gib=a.workspace_gib, device=a.device)
        common = {'selection': selection, 'executable': str(a.executable), 'device': a.device}
        self.stage('gpu_int8_regressions', 'gpu_test', executable=str(a.executable), engine=built['engine'], device=a.device,
                   image=read(selection)['parity'][0]['path'])
        models = {
            'pytorch': ('python', str(a.weights), 'fp32'),
            'onnx-int8': ('python', quantized['onnx'], 'int8'),
            'python-int8': ('python', built['engine'], 'int8'),
            'python-fp16': ('python', str(a.fp16_engine), 'fp16'),
            'native-int8': ('native-trt', built['engine'], 'int8'),
            'fused-int8': ('fused-trt', built['engine'], 'int8'),
            'native-fp16': ('native-trt', str(a.fp16_engine), 'fp16'),
            'fused-fp16': ('fused-trt', str(a.fp16_engine), 'fp16'),
            'raw-native-int8': ('native-trt', built['raw_engine'], 'int8'),
            'raw-fused-int8': ('fused-trt', built['raw_engine'], 'int8')}
        predictions = {}
        for label, (backend, model, precision) in models.items():
            predictions[label] = self.stage('parity_' + label, 'predict', **common, backend=backend,
                model=model, precision=precision, label=label, split='parity', confidence=.25)['chunks']
        pairs = [('native-int8', 'python-int8', True), ('fused-int8', 'python-int8', True),
                 ('fused-int8', 'native-int8', True), ('raw-native-int8', 'native-int8', True),
                 ('raw-fused-int8', 'fused-int8', True), ('native-fp16', 'python-fp16', True),
                 ('fused-fp16', 'native-fp16', True), ('native-int8', 'native-fp16', False),
                 ('python-int8', 'pytorch', False), ('onnx-int8', 'pytorch', False)]
        for candidate, reference, same in pairs:
            self.stage(f'compare_{candidate}_vs_{reference}', 'compare', candidate=predictions[candidate],
                       reference=predictions[reference], same_artifact=same)
        for label, (backend, model, precision) in models.items():
            if label.startswith('raw-'):
                continue
            result = self.stage('quality_predict_' + label, 'predict', **common, backend=backend,
                model=model, precision=precision, label=label, split='quality', confidence=.001)
            self.stage('quality_' + label, 'score', selection=selection, chunks=result['chunks'], smoke=a.smoke)
        configurations = [(precision, backend, mode, depth) for precision in ('fp16', 'int8')
            for backend in ('native-trt', 'fused-trt')
            for mode, depth in (('latency', 1), ('throughput', 1), ('throughput', 3))]
        for repetition in range(1, (1 if a.smoke else 3) + 1):
            for precision, backend, mode, depth in configurations[::1 if repetition % 2 else -1]:
                measurement = self.stage(f'benchmark_{precision}_{backend}_{mode}_d{depth}_r{repetition}', 'benchmark',
                    **common, precision=precision, backend=backend, mode=mode, depth=depth, smoke=a.smoke,
                    model=str(a.fp16_engine) if precision == 'fp16' else built['engine'])
                semantics = measurement['semantics']
                if self.manifest.setdefault('benchmark_semantics', semantics) != semantics:
                    raise ValueError('Unmatched input/preprocessing/postprocessing/device/graph semantics')
        if fingerprint(a) != self.manifest['identity']:
            raise RuntimeError('Source/input/runtime identity changed; results are not validated')
        # Input hashes recorded by preparation cover labels too, including any
        # source hardlinks used by prediction workers.
        selected = read(selection)
        for group in ('calibration', 'parity', 'quality', 'benchmark'):
            for row in selected[group]:
                verify_outputs([row, row['label']])
        self.manifest.update(status='completed', evidence_validated=True, finished_utc=utc_now())
        self.save()
        return 0


def render(output, manifest):
    stages = manifest['stages']
    def result(name):
        row = stages.get(name, {})
        return row.get('result') if row.get('status') == 'completed' else None
    lines = ['# INT8 deployment evaluation', '', f"Status: **{manifest['status']}**.", '']
    if manifest['diagnostic_only']:
        lines += ['**SMOKE ONLY — no full calibration, full-test mAP or publishable performance claim.**', '']
    if not manifest.get('evidence_validated'):
        lines += ['Evidence is provisional/unverified until the complete run validates identities.', '']
    lines += ['## Quality (0–100 scale)', '', '| Pipeline | mAP50 | mAP50–95 | Images |', '|---|---:|---:|---:|']
    quality = {}
    for label in ('pytorch', 'onnx-int8', 'python-int8', 'python-fp16', 'native-int8', 'fused-int8', 'native-fp16', 'fused-fp16'):
        row = result('quality_' + label)
        if row:
            quality[label] = row
            lines.append(f"| {label} | {100 * row['metrics/mAP50(B)']:.4f} | {100 * row['metrics/mAP50-95(B)']:.4f} | {row['images']} |")
        else:
            lines.append(f'| {label} | PENDING | PENDING | — |')
    gates = []
    limit = manifest['identity']['settings']['max_map_drop']
    for label in ('native-int8', 'fused-int8'):
        base = label.replace('int8', 'fp16')
        if label in quality and base in quality:
            drop = 100 * (quality[base]['metrics/mAP50-95(B)'] - quality[label]['metrics/mAP50-95(B)'])
            gates.append({'candidate': label, 'reference': base, 'drop_pp': drop,
                          'limit_pp': limit, 'passed': drop <= limit})
            lines += ['', f"{label}: mAP50–95 loss vs {base}: **{drop:+.4f} percentage points**; configured budget {limit:g} pp: **{'PASS' if drop <= limit else 'FAIL'}**."]
    lines += ['', 'The accuracy budget is an explicit engineering acceptance criterion, not a promise of lossless INT8.',
              '', '## Strict parity', '', '| Comparison | Matched images | Gate |', '|---|---:|---|']
    parity_failures = []
    for name, stage in stages.items():
        if name.startswith('compare_') and stage['status'] == 'completed':
            row = stage['result']
            lines.append(f"| {name[8:]} | {row['matched_images']}/{row['images']} | {'PASS' if row['passed'] else 'FAIL'} |")
            if row['same_artifact'] and not row['passed']:
                parity_failures.append(name)
    lines += ['', 'Cross-precision/PyTorch strict failures are reported, not disguised as same-artifact failures or hidden by mAP.',
              '', '## Performance', '', '| Precision / backend | Mode | Depth | Mean / p50 / p95 ms | FPS | FPS min–max / CV | Runs |',
              '|---|---|---:|---:|---:|---:|---:|']
    aggregates = {}
    required = 1 if manifest['diagnostic_only'] else 3
    for precision in ('fp16', 'int8'):
        for backend in ('native-trt', 'fused-trt'):
            for mode, depth in (('latency', 1), ('throughput', 1), ('throughput', 3)):
                key = f'{precision}_{backend}_{mode}_d{depth}'
                values = [result(f'benchmark_{key}_r{r}') for r in range(1, required + 1)]
                good = [x for x in values if x]
                aggregates[key] = {metric: variability([x[metric] for x in good]) for metric in ('mean_ms', 'p50_ms', 'p95_ms', 'fps')}
                if len(good) == required:
                    v = aggregates[key]
                    times = ' / '.join(f"{v[m]['median']:.3f}" for m in ('mean_ms', 'p50_ms', 'p95_ms'))
                    fps = v['fps']
                    cv = f"{fps['cv_percent']:.2f}%" if fps['cv_percent'] is not None else 'N/A'
                    lines.append(f"| {precision} / {backend} | {mode} | {depth} | {times} | {fps['median']:.3f} | {fps['minimum']:.3f}–{fps['maximum']:.3f} / {cv} | {len(good)}/{required} |")
                else:
                    lines.append(f'| {precision} / {backend} | {mode} | {depth} | PENDING | PENDING | — | {len(good)}/{required} |')
    lines += ['', 'Sequential latency and queue-inclusive streaming residence time are different measurements. Headline values are medians of run statistics; FPS uses measured wall time including draining.',
              '', '## INT8 versus FP16', '']
    for backend in ('native-trt', 'fused-trt'):
        for mode, depth in (('latency', 1), ('throughput', 1), ('throughput', 3)):
            metric = 'mean_ms' if mode == 'latency' else 'fps'
            fp = aggregates[f'fp16_{backend}_{mode}_d{depth}'][metric]
            iq = aggregates[f'int8_{backend}_{mode}_d{depth}'][metric]
            if fp and iq:
                ratio = fp['median'] / iq['median'] if mode == 'latency' else iq['median'] / fp['median']
                lines.append(f'- {backend}, {mode}, depth {depth}: INT8 speed ratio **{ratio:.3f}×** (>1 is faster).')
    lines += ['', 'Same decoded inputs, batch 1, 640×640, confidence 0.25, stage timers off, graph capture requested/fail-on-capture-error. Native/fused processing preserves TASK 1 semantics. INT8 may retain floating layers and FP32 external I/O; it is not an all-INT8 pipeline.',
              'Full protocol: 128 images, 50 warmups, 300 sequential or 1,000 streaming frames, three repetitions. Model loading/decode/drawing/disk are excluded. Cells run serially and order reverses between repetitions.',
              'Quality uses confidence 0.001, max 300 detections and square preprocessing for every backend, with Ultralytics matching/AP on original-image boxes. Full quality includes C++ native and fused outputs, not only Python artifacts.',
              'Full-run acceptance requires every same-artifact parity gate and the configured accuracy budget. Faster timings do not override accuracy failures.', '']
    if manifest.get('error'):
        lines += [f"Error: {manifest['error']}", '']
    if any(stage['spec'].get('reuse_engine') for stage in stages.values()):
        lines += ['The TensorRT INT8 engine was reused. Its calibration is inherited; it was NOT built from this run’s ONNX/sample.', '']
    summary = {'status': manifest['status'], 'diagnostic_only': manifest['diagnostic_only'],
               'evidence_validated': manifest.get('evidence_validated', False), 'quality': quality,
               'accuracy_gates': gates, 'same_artifact_parity_failures': parity_failures, 'performance': aggregates,
               'accepted': manifest['status'] == 'completed' and not manifest['diagnostic_only']
                    and manifest.get('evidence_validated', False) and not parity_failures
                    and len(gates) == 2 and all(x['passed'] for x in gates)}
    save_json(output / 'summary.json', summary)
    (output / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--reexport', action='store_true')
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/int8-deployment/bin/Release/edge_perception.exe')
    for name, default in (('weights', 'runs/train/yolo26m_bdd100k/weights/best.pt'),
                          ('data', 'data/processed/bdd100k_yolo/bdd100k.yaml'),
                          ('fp32-onnx', 'models/onnx/yolo26m_bdd100k_fp32.onnx'),
                          ('fp16-engine', 'models/engine/yolo26m_bdd100k_fp16.engine')):
        parser.add_argument('--' + name, type=Path, default=ROOT / default)
    for name in ('reuse-int8-engine', 'reuse-int8-onnx', 'qat-onnx', 'qat-provenance'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--calibration-images', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--workspace-gib', type=float, default=2)
    parser.add_argument('--max-map-drop', type=float, default=2.0)
    args = parser.parse_args()
    if args.calibration_images < 1 or args.workspace_gib <= 0 or args.max_map_drop < 0 or args.device < 0:
        parser.error('Invalid calibration count, workspace, accuracy budget or device')
    if bool(args.qat_onnx) != bool(args.qat_provenance) or (args.qat_onnx and args.reuse_int8_onnx):
        parser.error('QAT requires both provenance and ONNX, without a PTQ ONNX override')
    if args.smoke:
        if args.qat_onnx or args.reexport:
            parser.error('Smoke only reuses existing artifacts; no export or QAT')
        args.reuse_int8_engine = args.reuse_int8_engine or ROOT / 'models/engine/yolo26m_bdd100k_int8.engine'
        args.reuse_int8_onnx = args.reuse_int8_onnx or ROOT / 'models/onnx/yolo26m_bdd100k_allconv_int8.onnx'
    if args.reuse_int8_onnx and not args.reuse_int8_engine:
        # Asymmetric legacy QDQ must not be silently used as a TRT build source.
        from training.int8_workflow import validate_qdq
        validate_qdq(args.reuse_int8_onnx)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    if sys.platform == 'win32' and len(str(args.output.resolve())) > 100:
        parser.error('Use a shorter output directory (at most 100 characters) for native Windows report paths')
    pipeline = None
    try:
        with OutputLock(args.output):
            pipeline = Pipeline(args)
            code = pipeline.run()
            summary = read(args.output / 'summary.json')
            if summary['same_artifact_parity_failures'] or (not args.smoke and not summary['accepted']):
                print('COMPLETED WITH ACCEPTANCE FAILURES: inspect the report; no speed/quality claim is approved.')
                return 2
            print(f'FINISHED: {args.output / "README.md"}')
            return code
    except BaseException as error:
        if pipeline is not None:
            pipeline.manifest.update(status='failed', error=str(error), evidence_validated=False)
            pipeline.save()
        print(f'STOPPED: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
