"""Install hash-verified accepted INT8 artifacts, archiving previous active files."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_release import file_record, save_json, sha256, utc_now
from tools.int8_pipeline import read, verify_outputs


def promote(run: Path, models: Path):
    run, models = run.resolve(), models.resolve()
    manifest, summary = read(run / 'manifest.json'), read(run / 'summary.json')
    if (manifest['status'] != 'completed' or not manifest.get('evidence_validated')
            or manifest.get('diagnostic_only') or summary.get('accepted') is not True):
        raise ValueError('Only a completed, identity-validated, accepted full run can be promoted')
    for key in ('quantize', 'engine'):
        if manifest['stages'][key]['status'] != 'completed':
            raise ValueError('Missing completed artifact stage')
        verify_outputs(manifest['stages'][key]['outputs'])
    q = manifest['stages']['quantize']['result']
    e = manifest['stages']['engine']['result']
    targets = [(Path(q['onnx']), models / 'onnx/yolo26m_bdd100k_int8.onnx'),
               (Path(e['engine']), models / 'engine/yolo26m_bdd100k_int8.engine')]
    record_path = models / 'int8_active.json'
    if record_path.exists():
        old = read(record_path)
        if old['source_run'] == str(run) and all(dst.exists() and sha256(dst) == sha256(src) for src, dst in targets):
            verify_outputs(old['active_artifacts'])
            print('Validated INT8 artifacts are already active; no changes.')
            return old
        raise ValueError('Another promotion record exists; review it before replacing active artifacts')
    # This directory is outside onnx/ and engine/, so discovery cannot benchmark
    # archived files accidentally. Copy first; never remove original evidence.
    archive = models / 'archive' / ('int8_before_' + run.parent.name)
    archive.mkdir(parents=True, exist_ok=True)
    sidecars = [models / 'onnx/yolo26m_bdd100k_onnx_int8_metadata.json',
                models / 'engine/yolo26m_bdd100k_engine_int8_metadata.json']
    backups = []
    for path in [dst for _, dst in targets] + sidecars:
        if path.exists():
            backup = archive / path.relative_to(models)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists() and sha256(backup) != sha256(path):
                raise ValueError(f'Archive conflict: {backup}')
            if not backup.exists():
                shutil.copy2(path, backup)
            if sha256(path) != sha256(backup):
                raise ValueError(f'Backup verification failed: {backup}')
            backups.append(file_record(backup))
    record = {'source_run': str(run), 'source_manifest': file_record(run / 'manifest.json'),
              'source_summary': file_record(run / 'summary.json'), 'promoted_utc': utc_now(),
              'archived_files': backups, 'active_artifacts': [], 'policy': q['policy']}
    # Keep copies of the previous bytes recoverable even if interrupted mid-copy.
    for src, dst in targets:
        dst.parent.mkdir(parents=True, exist_ok=True)
        temporary = dst.with_suffix(dst.suffix + '.promoting')
        shutil.copy2(src, temporary)
        if sha256(src) != sha256(temporary):
            raise ValueError(f'Copy verification failed: {src}')
        temporary.replace(dst)
        record['active_artifacts'].append(file_record(dst))
    for sidecar, active in zip(sidecars, record['active_artifacts']):
        save_json(sidecar, {'schema_version': 1, 'artifact': active,
            'source_run': str(run), 'method': 'explicit_symmetric_qdq_mixed_precision_ptq',
            'policy': q['policy'], 'full_test_accepted': True,
            'note': 'Supersedes archived legacy metadata; see int8_active.json for provenance.'})
    save_json(record_path, record)
    print('Active INT8 artifacts verified. Previous files preserved at:', archive)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Completed pipeline directory')
    parser.add_argument('--models', type=Path, default=ROOT / 'models')
    args = parser.parse_args()
    promote(args.run, args.models)
