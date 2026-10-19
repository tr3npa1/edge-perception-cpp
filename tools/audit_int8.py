"""Inspect QDQ scales/placement and optionally probe a single real image."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def inspect(path: Path) -> dict:
    import numpy as np
    import onnx
    from onnx import numpy_helper
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    initializers = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    producers = {output: node for node in model.graph.node for output in node.output}
    quantized_ops = Counter()
    invalid, asymmetric, scale_records = [], [], {}
    for node in model.graph.node:
        if node.op_type not in ('QuantizeLinear', 'DequantizeLinear'):
            continue
        scale = initializers.get(node.input[1])
        if scale is None or not np.isfinite(scale).all() or (scale <= 0).any():
            invalid.append(node.name)
        else:
            scale_records[node.name] = {'minimum': float(scale.min()), 'maximum': float(scale.max()), 'shape': list(scale.shape)}
        zero = initializers.get(node.input[2]) if len(node.input) > 2 else None
        if zero is not None and np.any(zero != 0):
            asymmetric.append(node.name)
        if node.op_type == 'QuantizeLinear' and node.input[0] in producers:
            quantized_ops[producers[node.input[0]].op_type] += 1
    tails = []
    for output in model.graph.output:
        node = producers.get(output.name)
        if node is not None:
            tails.append({'output': output.name, 'producer': node.op_type, 'name': node.name,
                          'inputs': list(node.input), 'scale': scale_records.get(node.name),
                          'zero': initializers[node.input[2]].tolist() if node.op_type == 'DequantizeLinear' and len(node.input) > 2 else None})
    return {'path': str(path.resolve()), 'operators': dict(Counter(n.op_type for n in model.graph.node)),
            'quantized_activation_producers': dict(quantized_ops), 'invalid_scales': invalid,
            'nonzero_zero_points': len(asymmetric), 'outputs': tails}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', type=Path)
    args = parser.parse_args()
    result = []
    for path in args.models:
        item = inspect(path)
        if args.image:
            import numpy as np
            import onnxruntime as ort
            from training.export_model import YoloCalibrationDataReader
            session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
            tensor = YoloCalibrationDataReader([args.image], session.get_inputs()[0].name, 640).get_next()
            output = session.run(None, tensor)[0]
            item['probe'] = {'shape': list(output.shape), 'finite': bool(np.isfinite(output).all()),
                             'max_confidence': float(output[0, :, 4].max()),
                             'detections_at_025': int((output[0, :, 4] >= .25).sum()),
                             'unique_confidences': np.unique(output[0, :, 4]).tolist()[:25]}
            del session
        result.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
