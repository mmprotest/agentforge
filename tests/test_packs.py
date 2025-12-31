from importlib.util import find_spec
from pathlib import Path
import json
import zipfile

import pytest

from agentforge.packs.manager import (
    build_pack,
    install_pack,
    sign_pack,
    validate_manifest_schema,
    validate_pack,
    verify_pack,
)


def _rewrite_zip_with_manifest(zip_path: Path, manifest: dict) -> None:
    temp = zip_path.with_suffix(".tmp")
    with zipfile.ZipFile(zip_path) as original, zipfile.ZipFile(
        temp, "w", compression=zipfile.ZIP_DEFLATED
    ) as new_zip:
        for info in original.infolist():
            if info.filename == "manifest.json":
                continue
            new_zip.writestr(info, original.read(info.filename))
        new_zip.writestr("manifest.json", json.dumps(manifest, indent=2))
    zip_path.unlink()
    temp.rename(zip_path)


def test_pack_build_sign_verify_install(tmp_path: Path) -> None:
    source = tmp_path / "pack"
    source.mkdir()
    (source / "README.md").write_text("hello", encoding="utf-8")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "version": "1.0.0",
                "type": "tool_pack",
                "publisher": "test",
                "entrypoints": ["demo-tool"],
            }
        ),
        encoding="utf-8",
    )
    zip_path = tmp_path / "pack.zip"
    build_pack(source, zip_path)
    sign_pack(zip_path, "secret")
    assert verify_pack(zip_path, "secret")
    dest = tmp_path / "install"
    install_pack(zip_path, dest, key="secret")
    assert (dest / "demo" / "1.0.0" / "README.md").exists()


def test_pack_verify_fails_on_tamper(tmp_path: Path) -> None:
    source = tmp_path / "pack"
    source.mkdir()
    (source / "README.md").write_text("hello", encoding="utf-8")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "version": "1.0.0",
                "type": "tool_pack",
                "publisher": "test",
                "entrypoints": ["demo-tool"],
            }
        ),
        encoding="utf-8",
    )
    zip_path = tmp_path / "pack.zip"
    build_pack(source, zip_path)
    manifest = sign_pack(zip_path, "secret")
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    (extract_dir / "README.md").write_text("tampered", encoding="utf-8")
    tampered_zip = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in extract_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(extract_dir).as_posix())
    _rewrite_zip_with_manifest(tampered_zip, manifest)
    assert verify_pack(tampered_zip, "secret") is False


def test_pack_validate_detects_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "pack"
    source.mkdir()
    (source / "README.md").write_text("hello", encoding="utf-8")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "version": "1.0.0",
                "type": "tool_pack",
                "publisher": "test",
                "entrypoints": ["demo-tool"],
            }
        ),
        encoding="utf-8",
    )
    zip_path = tmp_path / "pack.zip"
    build_pack(source, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    manifest["files"][0]["sha256"] = "0" * 64
    _rewrite_zip_with_manifest(zip_path, manifest)
    result = validate_pack(zip_path)
    assert result["ok"] is False
    assert any("hash" in error for error in result["errors"])


def test_pack_build_manifest_hashes(tmp_path: Path) -> None:
    source = tmp_path / "pack"
    source.mkdir()
    (source / "data.txt").write_text("payload", encoding="utf-8")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "version": "1.0.0",
                "type": "mixed",
                "publisher": "test",
                "entrypoints": [],
            }
        ),
        encoding="utf-8",
    )
    zip_path = tmp_path / "pack.zip"
    manifest = build_pack(source, zip_path)
    assert manifest.files[0]["path"] == "data.txt"
    assert len(manifest.files[0]["sha256"]) == 64


def test_pack_validate_schema_missing_fields(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "manifest.json").write_text(
        json.dumps({"name": "demo", "files": {}}), encoding="utf-8"
    )
    result = validate_pack(pack_dir)
    assert result["ok"] is False
    if find_spec("jsonschema") is not None:
        assert any("required property" in error for error in result["errors"])
        assert any("files" in error for error in result["errors"])
    else:
        assert any("missing required field: spec_version" in error for error in result["errors"])
        assert any("files must be a list" in error for error in result["errors"])
        assert any(
            "Install agentforge[schema] for full schema validation." in warning
            for warning in result["warnings"]
        )


def test_pack_install_blocks_zip_slip(tmp_path: Path) -> None:
    zip_path = tmp_path / "evil.zip"
    safe_payload = "ok"
    safe_hash = __import__("hashlib").sha256(safe_payload.encode("utf-8")).hexdigest()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "spec_version": "0.1",
                    "name": "evil",
                    "version": "0.0.1",
                    "created_at": "2024-01-01T00:00:00Z",
                    "publisher": "test",
                    "type": "tool_pack",
                    "entrypoints": [],
                    "files": [{"path": "README.md", "sha256": safe_hash}],
                }
            ),
        )
        archive.writestr("README.md", safe_payload)
        archive.writestr("../evil.txt", "oops")
    dest = tmp_path / "install"
    try:
        install_pack(zip_path, dest, allow_unsigned=True)
    except RuntimeError as exc:
        assert "Unsafe path" in str(exc) or "Path traversal" in str(exc)
    else:
        raise AssertionError("Zip Slip not blocked")


def test_manifest_schema_validation_uses_jsonschema_when_available() -> None:
    pytest.importorskip("jsonschema")
    manifest = {
        "name": "demo",
        "version": "1.0.0",
        "created_at": "2024-01-01T00:00:00Z",
        "publisher": "test",
        "type": "tool_pack",
        "entrypoints": [],
        "files": [],
    }
    errors = validate_manifest_schema(manifest)
    assert any("spec_version" in error for error in errors)
