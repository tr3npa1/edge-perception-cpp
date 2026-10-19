"""Fresh all-artifact/backend evaluation; no Docker, training or publication writes."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_release import OutputLock, discover_artifacts, file_record, hardware_snapshot, save_json, sha256, utc_now
from tools.int8_pipeline import Pipeline, fingerprint, read, verify_outputs
from tools.benchmark_fp16_matrix import variability


def inventory(models, weights):
    promotion = read(models / 'int8_active.json')
    verify_outputs(promotion['active_artifacts'])
    # Verify the active copies, not merely the paths recorded during promotion.
    expected = {Path(x['path']).name: x['sha256'] for x in promotion['active_artifacts']}
    for kind, suffix in (('onnx', '.onnx'), ('engine', '.engine')):
        path = models / kind / ('yolo26m_bdd100k_int8' + suffix)
        if sha256(path) != expected[path.name]:
            raise ValueError(f'Active INT8 differs from accepted artifact: {path}')
    artifacts = discover_artifacts(models)
    required = {(kind, precision) for kind in ('onnx', 'engine') for precision in ('fp32', 'fp16', 'int8')}
    if not required <= {(a['artifact_type'], a['precision']) for a in artifacts}:
        raise ValueError('Need FP32, FP16 and INT8 artifacts for both ONNX and TensorRT')
    for a in artifacts:
        a['label'] = a['id'].replace('yolo26m_bdd100k_', '')
    if len({a['label'] for a in artifacts}) != len(artifacts):
        raise ValueError('Ambiguous artifact labels')
    return [{'label': 'pytorch', 'artifact_type': 'pytorch', 'precision': 'fp32',
             **file_record(weights), 'aliases': []}] + artifacts


def cells(artifacts):
    result = []
    for a in artifacts:
        if a['artifact_type'] == 'pytorch':
            continue
        backends = ('ort-cpu', 'ort-cuda', 'ort-trt') if a['artifact_type'] == 'onnx' else ('native-trt', 'fused-trt')
        for backend in backends:
            result.append({'label': a['label'] + '__' + backend, 'artifact': a['label'],
                           'backend': backend, 'model': a['path'], 'precision': a['precision'],
                           'supported': not (backend == 'ort-trt' and a['precision'] == 'int8')})
    return result


def identity(args, artifacts):
    return {**fingerprint(args), 'artifacts': artifacts,
            'launcher': file_record(ROOT / 'tools/run_all_models.ps1'),
            'promotion': file_record(args.models_dir / 'int8_active.json')}


class AllModels(Pipeline):
    worker = ROOT / 'tools/all_models_worker.py'

    def __init__(self, args):
        self.args, self.output = args, args.output.resolve()
        artifacts = inventory(args.models_dir, args.weights)
        current = identity(args, artifacts)
        path = self.output / 'manifest.json'
        if path.exists():
            if not args.resume:
                raise ValueError('Use a new directory or --resume')
            self.manifest = read(path)
            if self.manifest['identity'] != current:
                raise ValueError('Cannot resume: source, settings, artifacts or runtime changed')
        else:
            if args.resume:
                raise ValueError('No all-model manifest to resume')
            self.manifest = {'schema_version': 1, 'created_utc': utc_now(), 'identity': current,
                'hardware': hardware_snapshot(ROOT), 'stages': {}, 'diagnostic_only': args.smoke,
                'artifacts': artifacts, 'cells': cells(artifacts)}
        self.manifest.update(status='running', error=None, evidence_validated=False)
        self.save()

    def save(self):
        self.manifest['updated_utc'] = utc_now()
        save_json(self.output / 'manifest.json', self.manifest)
        render(self.output, self.manifest)

    def attempt(self, name, action, **payload):
        try:
            return self.stage(name, action, **payload)
        except RuntimeError as error:
            # A model/provider execution failure is evidence, not a reason to
            # erase other cells. Identity/checkpoint violations still abort.
            print(f'RECORDED FAILURE: {error}', flush=True)
            return None

    def run(self):
        a, m = self.args, self.manifest
        prepared = self.stage('selection', 'prepare', data=str(a.data), seed=42, smoke=a.smoke,
                              evaluation_only=True)
        selection = prepared['selection']
        frozen = self.output.parent / 'frames'
        frozen.mkdir(exist_ok=True)
        for row in read(selection)['benchmark']:
            dest = frozen / (row['key'] + Path(row['path']).suffix.lower())
            if not dest.exists():
                try:
                    os.link(row['path'], dest)
                except OSError:
                    shutil.copy2(row['path'], dest)
            if sha256(dest) != row['sha256']:
                raise ValueError('Frozen benchmark input changed')
        common = {'selection': selection, 'executable': str(a.executable), 'device': a.device}
        self.attempt('gpu_int8_regressions', 'gpu_test', executable=str(a.executable),
            engine=str(a.models_dir / 'engine/yolo26m_bdd100k_int8.engine'), device=a.device,
            image=read(selection)['parity'][0]['path'])
        references, predictions = {}, {}
        targets = []
        for artifact in m['artifacts']:
            label = 'python_' + artifact['label']
            targets.append({'label': label, 'backend': 'python', 'model': artifact['path'],
                            'precision': artifact['precision']})
            refs = self.attempt('parity_' + label, 'predict', **common, **targets[-1],
                                split='parity', confidence=.25)
            if refs:
                references[artifact['label']] = refs['chunks']
        for cell in m['cells']:
            if not cell['supported']:
                continue  # The performance smoke records the actual policy rejection.
            target = {key: cell[key] for key in ('label', 'backend', 'model', 'precision')}
            target['ort_cache'] = str(self.output.parent / 'quality_cache' / cell['artifact'])
            targets.append(target)
            pred = self.attempt('parity_' + cell['label'], 'predict', **common, **target,
                                split='parity', confidence=.25)
            if pred:
                predictions[cell['label']] = pred['chunks']
                for ref, same, suffix in ((cell['artifact'], True, 'same_artifact'), ('pytorch', False, 'pytorch')):
                    if ref in references:
                        self.attempt('compare_' + cell['label'] + '_' + suffix, 'compare',
                            candidate=pred['chunks'], reference=references[ref], same_artifact=same)
        for artifact, ref in references.items():
            if artifact != 'pytorch' and 'pytorch' in references:
                self.attempt('compare_python_' + artifact + '_pytorch', 'compare',
                    candidate=ref, reference=references['pytorch'], same_artifact=False)
        # Equal-artifact native/fused comparisons are explicit, not inferred
        # from each path independently passing against Python.
        for artifact in m['artifacts']:
            native, fused = artifact['label'] + '__native-trt', artifact['label'] + '__fused-trt'
            if native in predictions and fused in predictions:
                self.attempt('compare_' + fused + '_native', 'compare', candidate=predictions[fused],
                             reference=predictions[native], same_artifact=True)
        for target in targets:
            label = target['label']
            pred = self.attempt('quality_predict_' + label, 'predict', **common, **target,
                                split='quality', confidence=.001, chunk_timeout=7200)
            if pred:
                self.attempt('quality_' + label, 'score', selection=selection, chunks=pred['chunks'], smoke=a.smoke)
        self.attempt('performance', 'performance', executable=str(a.executable),
            models=str(a.models_dir), source=str(frozen), directory=str(self.output.parent / 'performance'),
            device=a.device, smoke=a.smoke)
        for repetition in range(1, (1 if a.smoke else 3) + 1):
            self.attempt(f'pytorch_timing_r{repetition}', 'pytorch_timing', selection=selection,
                         model=str(a.weights), device=a.device, smoke=a.smoke)
        if identity(a, inventory(a.models_dir, a.weights)) != m['identity']:
            raise ValueError('Run identity changed; results are unverified')
        for group in ('parity', 'quality', 'benchmark'):
            for row in read(selection)[group]:
                verify_outputs([row, row['label']])
        for stage in m['stages'].values():
            if stage['status'] == 'completed':
                verify_outputs(stage['outputs'])
        m.update(status='completed', evidence_validated=True, finished_utc=utc_now())
        self.save()
        print('ALL-MODEL SUITE FINISHED:', self.output / 'README.md', flush=True)
        return 2 if read(self.output / 'summary.json')['has_findings'] else 0


def render(output, m):
    def result(name):
        stage = m['stages'].get(name, {})
        return stage.get('result') if stage.get('status') == 'completed' else None
    lines = ['# All-model deployment evaluation', '', f"Status: **{m['status']}**; identity validated: **{m.get('evidence_validated', False)}**.", '']
    if m['diagnostic_only']:
        lines += ['**DIAGNOSTIC SMOKE ONLY — not publishable performance/full-test quality.**', '']
    lines += ['Fresh evidence only. No Docker, calibration, training, Git operation or public-doc overwrite.',
        'Complete does not mean every parity gate passed. Unsupported combinations and failures remain visible.', '',
        '## Artifact inventory', '', '| Artifact | SHA-256 | Aliases |', '|---|---|---|']
    for a in m['artifacts']:
        lines.append(f"| {a['label']} | `{a['sha256']}` | {', '.join(Path(x['path']).name for x in a['aliases']) or '—'} |")
    lines += ['', '## Quality', '', '| Path | Images | mAP50 | mAP50–95 |', '|---|---:|---:|---:|']
    quality = {}
    labels = ['python_' + a['label'] for a in m['artifacts']] + [c['label'] for c in m['cells'] if c['supported']]
    for label in labels:
        row = result('quality_' + label)
        if row:
            quality[label] = row
            lines.append(f"| {label} | {row['images']} | {row['metrics/mAP50(B)'] * 100:.4f} | {row['metrics/mAP50-95(B)'] * 100:.4f} |")
        else:
            failed = any(m['stages'].get(prefix + label, {}).get('status') == 'failed' for prefix in ('quality_', 'quality_predict_'))
            lines.append(f"| {label} | — | {'FAILED' if failed else 'PENDING'} | — |")
    lines += ['', 'Full mode evaluates the entire prepared test split for each Python artifact and each supported C++ path. '
        'Confidence 0.001, batch 1, square 640×640, max 300 detections; common matching/AP. Scores use a 0–100 scale.', '',
        '## Strict parity', '', '| Comparison | Images matched | Result |', '|---|---:|---|']
    parity = {}
    for name, stage in m['stages'].items():
        if name.startswith('compare_') and stage['status'] == 'completed':
            row = stage['result']
            parity[name] = row
            lines.append(f"| {name[8:]} | {row['matched_images']}/{row['images']} | {'PASS' if row['passed'] else 'FAIL'} |")
    lines += ['', 'Same-artifact gates and PyTorch/cross-precision gates are different. Failed gates are not waived because mAP is close.', '',
        '## C++ performance', '', '| Artifact / backend | Measurement | Depth | Mean / p50 / p95 ms | FPS | FPS min–max / CV | Runs |',
        '|---|---|---:|---:|---:|---:|---:|']
    perf = result('performance')
    performance = []
    if perf:
        pm = read(perf['manifest'])
        groups = {}
        for run in pm['runs']:
            if run['test'] not in ('smoke', 'stage_diagnostic'):
                groups.setdefault((run['artifact'], run['backend'], run['test'], run['depth']), []).append(run)
        required = 1 if m['diagnostic_only'] else 3
        for (artifact, backend, test, depth), rows in groups.items():
            good = [r for r in rows if r['status'] == 'passed']
            stats = {k: variability([r['metrics'][k] for r in good]) for k in ('mean_ms', 'median_ms', 'p95_ms', 'fps')}
            performance.append({'artifact': artifact, 'backend': backend, 'test': test, 'depth': depth,
                                'successful_runs': len(good), 'required_runs': required, 'statistics': stats})
            if len(good) == required:
                times = ' / '.join(f"{stats[k]['median']:.3f}" for k in ('mean_ms', 'median_ms', 'p95_ms'))
                fps = stats['fps']
                cv = f"{fps['cv_percent']:.2f}%" if fps['cv_percent'] is not None else 'N/A'
                lines.append(f"| {artifact} / {backend} | {test} | {depth} | {times} | {fps['median']:.3f} | {fps['minimum']:.3f}–{fps['maximum']:.3f} / {cv} | {len(good)}/{required} |")
            else:
                lines.append(f'| {artifact} / {backend} | {test} | {depth} | INCOMPLETE | — | — | {len(good)}/{required} |')
        lines += ['', '### Execution / unsupported cells', '']
        for cell in perf['cells']:
            lines.append(f"- {cell['cell_id']}: {cell['status']}. {cell.get('reason', '').splitlines()[0] if cell.get('reason') else ''}")
    else:
        lines.append('| All | PENDING or failed — inspect stage log | — | — | — | — | — |')
    pt = [result(f'pytorch_timing_r{i}') for i in range(1, (1 if m['diagnostic_only'] else 3) + 1)]
    pt_good = [x for x in pt if x]
    pt_stats = {k: variability([x[k] for x in pt_good]) for k in ('mean_ms', 'p50_ms', 'p95_ms', 'fps')}
    lines += ['', '## PyTorch reference timing', '']
    if len(pt_good) == len(pt):
        lines.append('Sequential mean/p50/p95: ' + ' / '.join(f"{pt_stats[k]['median']:.3f}" for k in ('mean_ms', 'p50_ms', 'p95_ms')) + f" ms; {pt_stats['fps']['median']:.3f} FPS.")
    else:
        lines.append('PENDING or failed — inspect stage log.')
    lines += ['Predecoded BGR through the Python API to Results, synchronized per frame; includes Python overhead. Not a C++ backend or depth-3 streaming claim.', '',
        '## Methodology and boundaries', '',
        '- 128 identical frozen images; 50 warmups, 300 sequential / 1,000 streaming frames, three repetitions. Smoke uses 3 images, 3 warmups, 6 frames and one repetition.',
        '- C++ sequential end-to-end latency excludes decode/setup/drawing/disk. Streaming frame residence is different; FPS uses completed frames / wall time including draining.',
        '- Native/fused TRT cover FP32/FP16/INT8 at depths 1 and 3; fused depth 2 is also retained. ORT streaming is explicitly depth 1. Backend-only scopes and instrumented stage diagnostics are separate.',
        '- Fused FP16 graph-on/off ablations are retained. Native depth 3 is checked for real worker concurrency; requested fused graph capture cannot silently fall back.',
        '- ORT TRT may partition across TRT/CUDA/CPU. INT8 QDQ through ORT TRT is rejected by project policy; the performance smoke records the rejection, not fabricated timings.',
        '- Cell order reverses between repetitions. No GPU clock locking or statistical significance claim. Compare native/fused and INT8/FP16 at equal depth within this run.',
        '- Current canonical INT8 replaces the old broken export; archived old bytes are not active benchmark inputs. The legacy all-convolution ONNX remains a separately labeled comparison.',
        '- Full predictions/checkpoints, raw per-frame timings, stage diagnostics, annotated smoke outputs, commands and errors are retained locally. No full annotated-image duplication.', '']
    failures = [name for name, stage in m['stages'].items() if stage['status'] == 'failed']
    findings = bool(failures or any(not row['passed'] for row in parity.values()) or (perf and perf['failed_attempts']))
    if failures:
        lines += ['## Failed jobs', ''] + ['- ' + name for name in failures]
    if m.get('error'):
        lines += ['', 'Error: ' + m['error']]
    save_json(output / 'summary.json', {'status': m['status'], 'diagnostic_only': m['diagnostic_only'],
        'evidence_validated': m.get('evidence_validated', False), 'quality': quality, 'parity': parity,
        'performance': performance, 'pytorch_timing': pt_stats, 'failed_stages': failures, 'has_findings': findings})
    (output / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (('models-dir', 'models'), ('weights', 'runs/train/yolo26m_bdd100k/weights/best.pt'),
                          ('data', 'data/processed/bdd100k_yolo/bdd100k.yaml'),
                          ('executable', 'build/int8-deployment/bin/Release/edge_perception.exe')):
        parser.add_argument('--' + name, type=Path, default=ROOT / default)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.device < 0 or len(str(args.output)) > 90:
        parser.error('Need nonnegative device and an output path at most 90 characters long')
    artifacts = inventory(args.models_dir, args.weights)
    if not args.data.is_file():
        parser.error('Dataset YAML is missing')
    if args.check_only:
        print(f'Validated inventory: {len(artifacts)} unique artifacts including PyTorch; {len(cells(artifacts))} C++ combinations.')
        for artifact in artifacts:
            print(artifact['label'], artifact['path'])
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    pipeline = None
    try:
        with OutputLock(args.output):
            pipeline = AllModels(args)
            return pipeline.run()
    except BaseException as error:
        if pipeline:
            pipeline.manifest.update(status='failed', error=str(error), evidence_validated=False)
            pipeline.save()
        print('STOPPED:', error, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
