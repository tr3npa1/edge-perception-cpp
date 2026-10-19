"""Explicit, symmetric QDQ PTQ for the fixed YOLO26 deployment contract.

This module never runs training. QAT mode consumes an independently trained,
provenance-backed QDQ ONNX artifact; it does not relabel PTQ as QAT.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import random
import re


def select_calibration(data: Path, count: int, seed: int) -> tuple[list[Path], dict]:
    import cv2
    from training.evaluate import collect_images, validate_dataset_yaml
    dataset = validate_dataset_yaml(data, 'train')
    images = collect_images(dataset.split_image_roots)
    if len(images) < count:
        raise ValueError(f'Need {count} training images; found {len(images)}')
    # Sample across the training split, then cover class presence and brightness
    # strata. This is an auditable proxy for diversity, not weather annotation.
    pool = random.Random(seed).sample(images, min(len(images), max(count * 4, 64)))
    rows = []
    for path in pool:
        gray = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if gray is None:
            raise ValueError(f'Unreadable calibration image: {path}')
        label = path.parent.parent / 'labels' / (path.stem + '.txt')
        classes = {int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()}
        if not classes <= set(range(dataset.class_count)):
            raise ValueError(f'Invalid calibration class: {label}')
        rows.append((path, min(3, int(float(gray.mean()) / 64)), classes))
    selected = []
    for cls in range(dataset.class_count):
        match = next((row for row in rows if cls in row[2] and row not in selected), None)
        if match and len(selected) < count:
            selected.append(match)
    buckets = [[row for row in rows if row[1] == bucket and row not in selected] for bucket in range(4)]
    while len(selected) < count:
        for bucket in buckets:
            if bucket and len(selected) < count:
                selected.append(bucket.pop())
    return [row[0] for row in selected], {
        'split': 'train', 'seed': seed, 'candidate_pool': len(pool),
        'selection': 'seeded candidate pool; class-presence coverage then brightness-stratified sampling',
        'brightness_bins': dict(Counter(row[1] for row in selected)),
        'class_image_counts': dict(Counter(cls for row in selected for cls in row[2])),
        'missing_classes': [cls for cls in range(dataset.class_count) if not any(cls in row[2] for row in selected)]}


def quantization_policy(model) -> dict:
    # The output producer identifies the detection module in this exported YOLO
    # family. Refuse an unknown graph instead of accidentally quantizing its head.
    producers = {out: node for node in model.graph.node for out in node.output}
    output = producers[model.graph.output[0].name]
    match = re.match(r'(/model\.\d+)/', output.name)
    if not match or output.op_type != 'Concat':
        raise ValueError('Unknown YOLO detection head; explicit exclusion policy is required')
    prefix = match.group(1) + '/'
    selected = [node.name for node in model.graph.node if node.op_type == 'Conv' and not node.name.startswith(prefix)]
    excluded = [node.name for node in model.graph.node if node.op_type == 'Conv' and node.name.startswith(prefix)]
    if not selected or not excluded or len(set(selected)) != len(selected):
        raise ValueError('Cannot identify unique backbone/neck Conv nodes and excluded head')
    return {'nodes_to_quantize': selected, 'nodes_to_exclude': excluded,
            'floating_head_prefix': prefix, 'operators': ['Conv'],
            'activation': 'symmetric INT8 per-tensor', 'weight': 'symmetric INT8 per-channel',
            'bias': 'floating', 'calibration': 'MinMax', 'external_io': 'FP32'}


def validate_qdq(path: Path, *, trt_compatible: bool = True) -> dict:
    from tools.audit_int8 import inspect
    info = inspect(path)
    if not info['operators'].get('QuantizeLinear') or not info['operators'].get('DequantizeLinear'):
        raise ValueError('Expected a genuinely QDQ-quantized artifact')
    if info['invalid_scales']:
        raise ValueError('Nonpositive/nonfinite/missing quantization scales')
    if any(row['producer'] in ('QuantizeLinear', 'DequantizeLinear') for row in info['outputs']):
        raise ValueError('Mixed-unit detector output must not be quantized')
    if trt_compatible and info['nonzero_zero_points']:
        raise ValueError('TensorRT QDQ requires symmetric zero-points; no automatic graph rewriting')
    import onnx
    model = onnx.load(str(path))
    for value, expected in ((model.graph.input[0], [1, 3, 640, 640]),
                            (model.graph.output[0], [1, 300, 6])):
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError('This workflow requires FP32 external QDQ input/output')
        if [d.dim_value for d in value.type.tensor_type.shape.dim] != expected:
            raise ValueError(f'Wrong deployment shape: {value.name}')
    return info


def quantize(source: Path, output: Path, images: list[Path]) -> dict:
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from training.export_model import YoloCalibrationDataReader
    model = onnx.load(str(source))
    policy = quantization_policy(model)
    quantize_static(str(source), str(output),
        YoloCalibrationDataReader(images, model.graph.input[0].name, 640),
        quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        per_channel=True, calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=['Conv'], nodes_to_quantize=policy['nodes_to_quantize'],
        nodes_to_exclude=policy['nodes_to_exclude'],
        extra_options={'ActivationSymmetric': True, 'WeightSymmetric': True, 'QuantizeBias': False})
    return {'policy': policy, 'graph': validate_qdq(output)}


def engine_bytes(path: Path) -> tuple[bytes, dict | None]:
    data = path.read_bytes()
    if len(data) > 4 and data[4:].lstrip().startswith(b'{'):
        length = int.from_bytes(data[:4], 'little')
        if not 0 < length <= 16 * 1024 * 1024 or 4 + length >= len(data):
            raise ValueError('Invalid engine metadata length or missing plan')
        metadata = json.loads(data[4:4 + length])
        if not isinstance(metadata, dict):
            raise ValueError('Engine metadata must be an object')
        return data[4 + length:], metadata
    if not data:
        raise ValueError('Empty engine')
    return data, None


def build_engine(onnx_path: Path, output: Path, names: dict, workspace_gib: float, device: int) -> dict:
    import torch
    import tensorrt as trt
    validate_qdq(onnx_path)
    if int(trt.__version__.split('.')[0]) != 10:
        raise RuntimeError('This tested builder targets TensorRT 10.x; no silent precision-policy migration')
    torch.cuda.set_device(device)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30)))
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # Explicit QDQ already carries the INT8 scales. No implicit INT8 flag or
    # second calibrator: unquantized regions may use FP16, preserving FP32 I/O.
    config.set_flag(trt.BuilderFlag.FP16)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError('TensorRT failed to build the explicitly quantized engine')
    metadata = {'task': 'detect', 'imgsz': [640, 640], 'batch': 1, 'stride': 32,
                'names': names, 'end2end': True, 'args': {'int8': True, 'half': False},
                'description': 'YOLO26 mixed INT8 backbone/neck, floating detection head'}
    encoded = json.dumps(metadata).encode('utf-8')
    output.write_bytes(len(encoded).to_bytes(4, 'little') + encoded + bytes(plan))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(bytes(plan))
    if engine is None:
        raise RuntimeError('Built engine failed deserialization')
    inspector = engine.create_engine_inspector()
    layers = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    output.with_suffix('.layers.json').write_text(layers, encoding='utf-8')
    return {'tensorrt': trt.__version__, 'device': torch.cuda.get_device_name(device),
            'workspace_gib': workspace_gib, 'explicit_qdq': True, 'fp16_unquantized_regions': True,
            'layer_inspector': str(output.with_suffix('.layers.json'))}
