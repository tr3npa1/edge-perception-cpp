"""Focused checks for documentation export; no model execution."""
import hashlib
import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("export_results", Path(__file__).with_name("export_results.py"))
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def test_scrub_keeps_repository_identity_but_removes_machine_data():
    value = {"path": str(exporter.ROOT / "src/main.cpp"), "uuid": "secret", "node": "host",
             "nested": ["C:\\Users\\person\\private.txt", "GPU-aabbcc-1122"], "sha256": "abc123"}
    cleaned = exporter.scrub(value)
    assert cleaned == {"path": "src/main.cpp", "nested": ["<external-path>", "<gpu-id>"], "sha256": "abc123"}


def test_table_escapes_cell_delimiters():
    assert "a\\|b" in exporter.table(["Name"], [["a|b"]])


def test_digest_uses_original_bytes(tmp_path):
    path = tmp_path / "evidence.bin"
    path.write_bytes(b"unchanged evidence")
    assert exporter.digest(path) == hashlib.sha256(b"unchanged evidence").hexdigest()


def test_published_text_has_stable_utf8_lf_bytes(tmp_path):
    path = tmp_path / "report.md"
    exporter.write_text(path, "Result: 640\u00d7640\nComplete\n")
    assert path.read_bytes() == "Result: 640\u00d7640\nComplete\n".encode("utf-8")


def test_csv_uses_lf_without_blank_records(tmp_path):
    rows = [{"name": "native,trt", "value": 3}, {"name": "fused", "value": 1}]
    path = tmp_path / "results.csv"
    exporter.write_text(path, exporter.csv_text(rows))
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert list(csv.DictReader(io.StringIO(raw.decode("utf-8")))) == [
        {"name": "native,trt", "value": "3"}, {"name": "fused", "value": "1"}]


@pytest.mark.parametrize("status,validated,smoke", [("running", False, False), ("completed", False, False), ("completed", True, True)])
def test_rejects_incomplete_unvalidated_or_smoke_run(tmp_path, status, validated, smoke):
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "performance").mkdir()
    manifest = {"status": status, "evidence_validated": validated, "diagnostic_only": smoke}
    for path, data in (("pipeline/manifest.json", manifest), ("pipeline/summary.json", {}), ("performance/manifest.json", {})):
        (tmp_path / path).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="validated, completed full run"):
        exporter.validate(tmp_path)


def test_refuses_existing_output_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "validate", lambda _: ({}, {}, {}, 0))
    with pytest.raises(ValueError, match="Output exists"):
        exporter.export(tmp_path, tmp_path)


def test_require_does_not_hide_validation_failure():
    with pytest.raises(ValueError, match="bad evidence"):
        exporter.require(False, "bad evidence")
