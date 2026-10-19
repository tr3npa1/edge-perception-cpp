"""Isolated, checkpointed jobs for run_int8_pipeline.ps1 (no implicit training)."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_release import file_record, save_json, sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def checked_predictions(rows, names):
    for row in rows:
        box = row['xyxy']
        if (row['class_id'] not in names or len(box) != 4 or not all(map(math.isfinite, box))
                or not math.isfinite(row['confidence']) or not 0 <= row['confidence'] <= 1
                or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError('Invalid detection emitted by backend')
    return rows


def source_records(data: Path, split: str, limit: int, seed: int):
    import random
    from training.evaluate import collect_images, validate_dataset_yaml
    dataset = validate_dataset_yaml(data, split)
    paths = collect_images(dataset.split_image_roots)
    if limit:
        if len(paths) < limit:
            raise ValueError(f'Need {limit} images in {split}')
        paths = random.Random(seed).sample(paths, limit)
    records = []
    for i, path in enumerate(paths):
        label = path.parent.parent / 'labels' / (path.stem + '.txt')
        records.append({'key': f'{i:06d}', **file_record(path), 'label': file_record(label)})
    return records


def verify_record(record):
    if sha256(Path(record['path'])) != record['sha256']:
        raise ValueError(f"Input changed: {record['path']}")


def prepare(job, output):
    import yaml
    from training.int8_workflow import select_calibration
    data = Path(job['data'])
    names = yaml.safe_load(data.read_text())['names']
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    # Smoke does not calibrate: do not enumerate/sample the full training split.
    calibration, diversity = ([], {'status': 'not_requested'}) if job['smoke'] or job.get('evaluation_only') else select_calibration(
        data, job['calibration_images'], job['seed'])
    result = {'names': names, 'calibration': [{**file_record(p), 'label': file_record(
        p.parent.parent / 'labels' / (p.stem + '.txt'))} for p in calibration], 'diversity': diversity,
        'parity': source_records(data, 'val', 4 if job['smoke'] else 64, job['seed']),
        'quality': source_records(data, 'test', 4 if job['smoke'] else 0, job['seed']),
        'benchmark': source_records(data, 'val', 3 if job['smoke'] else 128, job['seed'])}
    # Calibration never includes validation/test data, including aliases.
    held_out = {x['sha256'] for key in ('parity', 'quality', 'benchmark') for x in result[key]}
    if any(x['sha256'] in held_out for x in result['calibration']):
        raise ValueError('Calibration/held-out content overlap detected')
    save_json(output / 'selection.json', result)
    return {'selection': str(output / 'selection.json'), 'calibration_images': len(calibration),
            'quality_images': len(result['quality']), 'diversity': diversity}


def export(job, output):
    target = output / 'fp32.onnx'
    if job.get('reexport'):
        from ultralytics import YOLO
        checkpoint = output / 'source.pt'
        shutil.copy2(job['weights'], checkpoint)
        path = Path(YOLO(str(checkpoint)).export(format='onnx', imgsz=640, batch=1, dynamic=False,
            half=False, simplify=True, opset=18, device=job['device']))
        shutil.copy2(path, target)
    else:
        shutil.copy2(job['fp32_onnx'], target)
    import onnx
    from training.int8_workflow import quantization_policy
    model = onnx.load(str(target))
    onnx.checker.check_model(model)
    return {'onnx': str(target), 'reused': not job.get('reexport'), 'policy': quantization_policy(model)}


def quantize(job, output):
    from training.int8_workflow import quantize as ptq, validate_qdq
    target = output / 'mixed_int8.onnx'
    if job.get('qat_onnx'):
        provenance = read(job['qat_provenance'])
        if (provenance.get('method') != 'qat' or provenance.get('onnx_sha256') != sha256(Path(job['qat_onnx']))
                or provenance.get('base_checkpoint_sha256') != sha256(Path(job['weights']))
                or not provenance.get('training_command') or not provenance.get('training_evidence')):
            raise ValueError('QAT requires declared training evidence, command, ONNX and base-checkpoint hashes')
        shutil.copy2(job['qat_onnx'], target)
        info = {'mode': 'externally_trained_qat', 'provenance': provenance, 'graph': validate_qdq(target)}
    elif job.get('reuse_onnx'):
        shutil.copy2(job['reuse_onnx'], target)
        info = {'mode': 'reused_onnx', 'graph': validate_qdq(target, trt_compatible=False)}
    else:
        selection = read(job['selection'])
        for row in selection['calibration']:
            verify_record(row)
            verify_record(row['label'])
        info = ptq(Path(job['onnx']), target, [Path(row['path']) for row in selection['calibration']])
    return {'onnx': str(target), **info}


def engine(job, output):
    from training.int8_workflow import build_engine, engine_bytes
    target = output / 'model_int8.engine'
    if job.get('reuse_engine'):
        shutil.copy2(job['reuse_engine'], target)
        engine_bytes(target)  # Validate the envelope, actual execution follows.
        info = {'reused': True, 'original': file_record(Path(job['reuse_engine'])),
                'calibration': 'existing artifact; not claimed to use this run\'s new sample/ONNX'}
    else:
        info = build_engine(Path(job['onnx']), target, read(job['selection'])['names'],
                            job['workspace_gib'], job['device'])
    raw, metadata = engine_bytes(target)
    raw_path = output / 'model_int8_raw.engine'
    raw_path.write_bytes(raw)
    return {'engine': str(target), 'raw_engine': str(raw_path), 'metadata': metadata, **info}


def predict(job, output):
    """Checkpoint complete chunks; resume never trusts a partial chunk."""
    selection = read(job['selection'])
    images = selection[job['split']]
    names = {int(k): v for k, v in selection['names'].items()}
    checkpoint_path = output / 'chunks.json'
    checkpoints = read(checkpoint_path) if checkpoint_path.exists() else {}
    model = None
    if job['backend'] == 'python':
        from ultralytics import YOLO
        model = YOLO(job['model'], task='detect')
    chunk_size = 256
    for start in range(0, len(images), chunk_size):
        key = str(start)
        rows = images[start:start + chunk_size]
        for row in rows:
            verify_record(row)
            verify_record(row['label'])
        if key in checkpoints:
            verify_record(checkpoints[key])
            continue
        attempt = output / f'chunk_{start:06d}_{time.time_ns()}'
        attempt.mkdir()
        if model is not None:
            from training.evaluate import predict_one_image
            predictions = {row['key']: [asdict(d) for d in predict_one_image(model=model,
                image_path=Path(row['path']), imgsz=640, conf=job['confidence'], iou=.7,
                max_det=300, device=str(job['device']))] for row in rows}
        else:
            source = attempt / 'inputs'
            source.mkdir()
            for row in rows:
                path = Path(row['path'])
                staged = source / (row['key'] + path.suffix.lower())
                # Hardlinks avoid copying the whole test set; hashes are checked
                # before and after use. Neither CLI nor worker modifies inputs.
                try:
                    os.link(path, staged)
                except OSError:
                    shutil.copy2(path, staged)
            command = [job['executable'], '--mode', 'infer', '--source', str(source),
                '--model', job['model'], '--backend', job['backend'], '--precision', job['precision'],
                '--device', str(job['device']), '--confidence', str(job['confidence']),
                '--max-detections', '300', '--pipeline-depth', '1', '--json-only',
                '--output-dir', str(attempt / 'cpp')]
            if job.get('ort_cache'):
                command += ['--ort-cache-dir', job['ort_cache']]
            save_json(attempt / 'command.json', command)
            with (attempt / 'console.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=job.get('chunk_timeout', 1800))
            raw_rows = [json.loads(line) for line in (attempt / 'cpp/detections.jsonl').read_text().splitlines()]
            predictions = {Path(row['source']).stem: [{'class_id': d['class_id'], 'confidence': d['confidence'],
                'xyxy': d['box']} for d in row['detections']] for row in raw_rows}
            if len(raw_rows) != len(predictions):
                raise ValueError('Duplicate prediction records')
        if set(predictions) != {row['key'] for row in rows}:
            raise ValueError('Incomplete prediction coverage')
        for row in rows:
            checked_predictions(predictions[row['key']], names)
            verify_record(row)
        path = attempt / 'predictions.json'
        save_json(path, predictions)
        checkpoints[key] = file_record(path)
        save_json(checkpoint_path, checkpoints)
        print(f"{job['label']}: {min(start + chunk_size, len(images))}/{len(images)}", flush=True)
    return {'chunks': str(checkpoint_path), 'images': len(images), 'confidence': job['confidence']}


def score(job, output):
    import cv2
    import numpy as np
    import torch
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils.metrics import DetMetrics
    from ultralytics.utils.ops import xywh2xyxy
    from training.evaluate import to_jsonable
    selection = read(job['selection'])
    names = {int(k): v for k, v in selection['names'].items()}
    images = {row['key']: row for row in selection['quality']}
    validator = DetectionValidator(save_dir=output, args={'plots': False, 'device': 'cpu'})
    metrics = DetMetrics(names=names)
    seen, detections, targets = set(), 0, 0
    for chunk in read(job['chunks']).values():
        verify_record(chunk)
        for key, predictions in read(chunk['path']).items():
            if key in seen or key not in images:
                raise ValueError('Duplicate/unexpected quality image')
            seen.add(key)
            row = images[key]
            verify_record(row['label'])
            labels = np.array([list(map(float, line.split())) for line in Path(row['label']['path']).read_text().splitlines()
                               if line.strip()], dtype=np.float32).reshape(-1, 5)
            if (not np.isfinite(labels).all() or (labels[:, 1:] < 0).any() or (labels[:, 1:] > 1).any()
                    or any(c != int(c) or int(c) not in names for c in labels[:, 0])):
                raise ValueError(f"Invalid YOLO labels: {row['label']['path']}")
            image = cv2.imread(row['path'])
            if image is None:
                raise ValueError(f"Cannot decode {row['path']}")
            h, w = image.shape[:2]
            gt_boxes = xywh2xyxy(labels[:, 1:]) * np.array([w, h, w, h], dtype=np.float32)
            pred = {'bboxes': torch.tensor([d['xyxy'] for d in predictions], dtype=torch.float32).reshape(-1, 4),
                    'cls': torch.tensor([d['class_id'] for d in predictions], dtype=torch.float32)}
            gt = {'bboxes': torch.from_numpy(gt_boxes), 'cls': torch.from_numpy(labels[:, 0])}
            metrics.update_stats({**validator._process_batch(pred, gt),
                'conf': np.array([d['confidence'] for d in predictions]), 'pred_cls': pred['cls'].numpy(),
                'target_cls': labels[:, 0], 'target_img': np.unique(labels[:, 0]), 'im_name': key})
            detections += len(predictions)
            targets += len(labels)
    if seen != set(images):
        raise ValueError('Quality evaluation is incomplete')
    metrics.process(plot=False)
    return to_jsonable({'images': len(seen), 'detections': detections, 'targets': targets,
                       **metrics.results_dict, 'per_class': metrics.summary(),
                       'scope': 'small_sample_only' if job['smoke'] else 'full_prepared_test_split'})


def compare(job, output):
    from training.evaluate import Detection, compare_detection_sets
    from tools.check_fp16_parity import STRICT, PYTORCH
    def load(chunks):
        predictions = {}
        for record in read(chunks).values():
            verify_record(record)
            predictions.update(read(record['path']))
        return predictions
    reference, candidate = load(job['reference']), load(job['candidate'])
    if set(reference) != set(candidate) or not reference:
        raise ValueError('Parity inputs have different coverage')
    thresholds = STRICT if job['same_artifact'] else PYTORCH
    details = [asdict(compare_detection_sets(image_path=Path(key),
        reference_detections=[Detection(**d) for d in reference[key]],
        artifact_detections=[Detection(**d) for d in candidate[key]], **thresholds)) for key in reference]
    save_json(output / 'per_image.json', details)
    return {'matched_images': sum(d['passed'] for d in details), 'images': len(details),
            'passed': all(d['passed'] for d in details), 'thresholds': thresholds,
            'same_artifact': job['same_artifact'], 'details': str(output / 'per_image.json')}


def benchmark_semantics(target, backend):
    runtime = target['runtime_info']
    if backend == 'native-trt':
        config = target['engine_config']
        if target['type'] != 'inference_engine' or runtime['backend'] != 'native_tensorrt':
            raise ValueError('Unexpected native benchmark backend')
        return {'input_shape': runtime['model']['input_shape'],
                'preprocess': target['preprocess_config'], 'postprocess': target['postprocess_config'],
                'device': config['gpu']['device_id'], 'cuda_graph_requested': config['gpu']['enable_cuda_graph']}
    config = target['pipeline_config']
    if (target['type'] != 'native_tensorrt_pipeline' or not config['enable_gpu_postprocess']
            or not runtime['gpu_postprocess_enabled'] or not config['require_cuda_graph']):
        raise ValueError('Unexpected fused processing/graph policy')
    return {'input_shape': runtime['input_shape'], 'preprocess': config['preprocess'],
            'postprocess': config['postprocess'], 'device': config['device_id'],
            'cuda_graph_requested': config['enable_cuda_graph']}


def benchmark(job, output):
    from tools.benchmark_release import execute, read_benchmark
    from tools.benchmark_fp16_matrix import validate_report
    rows = read(job['selection'])['benchmark']
    source = output / 'inputs'
    source.mkdir(exist_ok=True)
    for row in rows:
        verify_record(row)
        target = source / (row['key'] + Path(row['path']).suffix)
        if not target.exists():
            shutil.copy2(row['path'], target)
        if sha256(target) != row['sha256']:
            raise ValueError('Benchmark staged image changed')
    folder = output / 'r'  # Parent is already unique for each stage attempt.
    iterations = (6 if job['smoke'] else 300) if job['mode'] == 'latency' else (6 if job['smoke'] else 1000)
    warmup = 3 if job['smoke'] else 50
    command = [job['executable'], '--mode', 'benchmark', '--source', str(source), '--model', job['model'],
        '--backend', job['backend'], '--precision', job['precision'], '--device', str(job['device']),
        '--pipeline-depth', str(job['depth']), '--output-dir', str(folder / 'app_output'),
        '--benchmark-mode', job['mode'], '--timing-scope', 'end-to-end', '--warmup', str(warmup),
        '--iterations', str(iterations), '--benchmark-source-frames', str(len(rows)), '--failure-policy', 'stop',
        '--confidence', '.25', '--max-detections', '300', '--retain-samples', '--no-stage-timings']
    if job['backend'] == 'fused-trt':
        command.append('--require-cuda-graph')
    process = execute(command, folder, ROOT, 1800)
    if process['exit_code'] != 0 or process['timed_out']:
        raise RuntimeError(f'Benchmark failed: {folder}')
    report, errors = read_benchmark(folder, iterations, warmup, job['mode'], 'end-to-end', job['depth'], False)
    errors += validate_report(report, job['backend'], job['mode'], job['depth'], len(rows), iterations)
    if job['backend'] == 'fused-trt' and any(sample.get('used_cuda_graph') is not True
                                           for sample in report.get('samples', [])):
        errors.append('A measured fused frame did not use the required CUDA graph')
    if errors:
        raise ValueError('; '.join(errors))
    for row in rows:
        verify_record(row)
    target = report['target']
    runtime = target['runtime_info']
    config = target['engine_config'] if job['backend'] == 'native-trt' else target['pipeline_config']
    actual_model = config.get('artifact_path', config.get('engine_path'))
    if Path(actual_model).resolve() != Path(job['model']).resolve():
        raise ValueError('Wrong benchmark model')
    if job['backend'] == 'native-trt' and runtime['artifact_precision'] != job['precision']:
        raise ValueError('Wrong benchmark precision')
    semantics = benchmark_semantics(target, job['backend'])
    if semantics['device'] != job['device'] or not semantics['cuda_graph_requested']:
        raise ValueError('Wrong device or graph configuration')
    return {'mean_ms': report['primary_latency']['mean_ms'], 'p50_ms': report['primary_latency']['median_ms'],
            'p95_ms': report['primary_latency']['p95_ms'], 'fps': report['throughput']['frames_per_second'],
            'target': target, 'semantics': semantics, 'maximum_active_workers': report.get('maximum_active_workers'),
            'raw_report': str(next(folder.rglob('benchmark.json')))}


def gpu_test(job, output):
    command = [str(Path(job['executable']).with_name('edge_gpu_correctness_tests.exe')),
               '--engine', job['engine'], '--image', job['image'], '--int8']
    save_json(output / 'command.json', command)
    with (output / 'console.log').open('w', encoding='utf-8') as log:
        # The regression executable uses device 0; remap only its process to
        # the selected physical device, leaving all benchmark processes alone.
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['device']))
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300, env=environment)
    return {'passed': True, 'command': command, 'cuda_visible_devices': str(job['device'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = read(args.job)
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true', MPLBACKEND='Agg')
    actions = {'prepare': prepare, 'export': export, 'quantize': quantize, 'engine': engine,
               'predict': predict, 'score': score, 'compare': compare, 'benchmark': benchmark, 'gpu_test': gpu_test}
    result = actions[job['action']](job, output)
    save_json(output / 'result.json', result)
    print(f"Completed {job['action']}", flush=True)


if __name__ == '__main__':
    main()
