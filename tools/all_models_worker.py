"""Isolated all-model jobs, reusing the checkpointed prediction/AP implementation."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import int8_worker as shared
from tools.benchmark_release import file_record, save_json


def performance(job, output):
    folder = Path(job['directory'])
    # An interruption before the initial manifest leaves no resumable work.
    # Preserve that directory and select a fresh attempt instead of deleting it.
    attempt = 1
    while folder.exists() and not (folder / 'manifest.json').exists():
        attempt += 1
        folder = Path(job['directory'] + f'_attempt{attempt}')
    command = [sys.executable, str(ROOT / 'tools/benchmark_release.py'),
        '--executable', job['executable'], '--output', str(folder), '--source', job['source'],
        '--models-dir', job['models'], '--device', str(job['device']),
        '--source-frames', '3' if job['smoke'] else '128',
        '--repetitions', '1' if job['smoke'] else '3', '--warmup', '3' if job['smoke'] else '50',
        '--latency-iterations', '6' if job['smoke'] else '300',
        '--throughput-iterations', '6' if job['smoke'] else '1000',
        '--stage-iterations', '3' if job['smoke'] else '100', '--timeout-seconds', '3600']
    if (folder / 'manifest.json').exists():
        command.append('--resume')
    save_json(output / 'command.json', command)
    code = subprocess.run(command, cwd=ROOT).returncode
    if code not in (0, 2):
        raise RuntimeError(f'Performance runner stopped with code {code}; evidence: {folder}')
    manifest = shared.read(folder / 'manifest.json')
    if manifest['status'] != 'completed':
        raise RuntimeError('Performance evidence is incomplete or invalidated')
    return {'manifest': str(folder / 'manifest.json'), 'aggregate': str(folder / 'aggregate/summary.json'),
            'cells': manifest['cells'], 'failed_attempts': manifest['failed_run_count'],
            'external_outputs': [file_record(p) for p in folder.rglob('*.json') if 'ort_cache' not in p.parts]}


def pytorch_timing(job, output):
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    rows = shared.read(job['selection'])['benchmark']
    frames = []
    for row in rows:
        shared.verify_record(row)
        frame = cv2.imread(row['path'])
        if frame is None:
            raise ValueError(f"Cannot decode benchmark input: {row['path']}")
        frames.append(frame)
    model = YOLO(job['model'], task='detect')
    warmup, count = (3, 6) if job['smoke'] else (50, 300)
    def run(index):
        model.predict(source=frames[index % len(frames)], imgsz=640, batch=1, device=str(job['device']),
            half=False, rect=False, conf=.25, iou=.7, max_det=300, verbose=False, save=False)
        torch.cuda.synchronize(job['device'])
    for i in range(warmup):
        run(i)
    samples = []
    start = time.perf_counter()
    for i in range(count):
        before = time.perf_counter()
        run(i)
        samples.append((time.perf_counter() - before) * 1000)
    elapsed = time.perf_counter() - start
    save_json(output / 'samples.json', samples)
    for row in rows:
        shared.verify_record(row)
    return {'mean_ms': float(np.mean(samples)), 'p50_ms': float(np.percentile(samples, 50)),
            'p95_ms': float(np.percentile(samples, 95)), 'fps': count / elapsed,
            'iterations': count, 'warmup': warmup, 'depth': 1,
            'scope': 'predecoded BGR to Python Results, includes Ultralytics API overhead, synchronized sequential',
            'precision': 'PyTorch FP32 request; not a C++ backend',
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = shared.read(args.job)
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true', MPLBACKEND='Agg')
    actions = {'prepare': shared.prepare, 'predict': shared.predict, 'score': shared.score,
               'compare': shared.compare, 'gpu_test': shared.gpu_test,
               'performance': performance, 'pytorch_timing': pytorch_timing}
    save_json(output / 'result.json', actions[job['action']](job, output))


if __name__ == '__main__':
    main()
