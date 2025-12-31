"""Pack build/sign/install helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import importlib.util
import json
from pathlib import Path
from posixpath import normpath
import sys
import zipfile
from typing import Any

from agentforge.packs.signing import sign_manifest, verify_manifest

SPEC_VERSION = "0.1"
ALLOWED_TYPES = {"workflow_pack", "tool_pack", "mixed"}
SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas/agent_pack_manifest_v0.1.schema.json"


@dataclass
class PackManifest:
    spec_version: str
    name: str
    version: str
    pack_type: str
    created_at: str
    publisher: str
    entrypoints: list[str]
    files: list[dict[str, str]]
    signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "spec_version": self.spec_version,
            "name": self.name,
            "version": self.version,
            "type": self.pack_type,
            "created_at": self.created_at,
            "publisher": self.publisher,
            "entrypoints": self.entrypoints,
            "files": self.files,
        }
        if self.signature:
            data["signature"] = self.signature
        return data


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(source_dir: Path) -> PackManifest:
    manifest_path = source_dir / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        name = payload.get("name") or source_dir.name
        version = payload.get("version") or "0.0.0"
        pack_type = payload.get("type") or "mixed"
        publisher = payload.get("publisher") or "unknown"
        entrypoints = payload.get("entrypoints") or []
    else:
        name = source_dir.name
        version = "0.0.0"
        pack_type = "mixed"
        publisher = "unknown"
        entrypoints = []
    if isinstance(entrypoints, dict):
        entrypoints = list(entrypoints.values())
    files: list[dict[str, str]] = []
    for path in source_dir.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(source_dir).as_posix()
        if relative == "manifest.json":
            continue
        files.append({"path": relative, "sha256": _hash_file(path)})
    files.sort(key=lambda item: item["path"])
    return PackManifest(
        spec_version=SPEC_VERSION,
        name=name,
        version=version,
        pack_type=pack_type,
        created_at=datetime.now(timezone.utc).isoformat(),
        publisher=publisher,
        entrypoints=entrypoints,
        files=files,
    )


def build_pack(source_dir: Path, output_path: Path) -> PackManifest:
    manifest = build_manifest(source_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest.to_dict(), indent=2))
        for file_info in manifest.files:
            archive.write(source_dir / file_info["path"], file_info["path"])
    return manifest


def read_manifest_from_zip(zip_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("manifest.json") as handle:
            return json.loads(handle.read().decode("utf-8"))


def sign_pack(pack_path: Path, key: str) -> dict[str, Any]:
    if pack_path.is_dir():
        manifest_path = pack_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["signature"] = sign_manifest(manifest, key)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    manifest = read_manifest_from_zip(pack_path)
    manifest["signature"] = sign_manifest(manifest, key)
    _rewrite_zip_manifest(pack_path, manifest)
    return manifest


def verify_pack(zip_path: Path, key: str) -> bool:
    manifest = read_manifest_from_zip(zip_path)
    if not verify_manifest(manifest, key):
        return False
    return verify_pack_files(zip_path, manifest)


def verify_pack_files(zip_path: Path, manifest: dict[str, Any]) -> bool:
    with zipfile.ZipFile(zip_path) as archive:
        for file_info in manifest.get("files", []):
            path = file_info["path"]
            expected = file_info["sha256"]
            with archive.open(path) as handle:
                digest = sha256(handle.read()).hexdigest()
            if digest != expected:
                return False
    return True


def install_pack(
    zip_path: Path,
    dest_root: Path,
    allow_unsigned: bool = False,
    key: str | None = None,
) -> Path:
    manifest = read_manifest_from_zip(zip_path)
    errors = validate_manifest_schema(manifest)
    if errors:
        raise RuntimeError("Manifest schema validation failed: " + "; ".join(errors))
    signature = manifest.get("signature")
    if signature and key:
        if not verify_manifest(manifest, key):
            raise RuntimeError("Pack signature verification failed")
    elif signature and not key:
        raise RuntimeError("Pack signature present but no key supplied")
    elif not signature and not allow_unsigned:
        raise RuntimeError("Unsigned pack blocked. Use --allow-unsigned to install.")
    if not verify_pack_files(zip_path, manifest):
        raise RuntimeError("Pack file hash verification failed")
    pack_dir = dest_root / manifest["name"] / manifest["version"]
    pack_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        safe_extract(archive, pack_dir)
    return pack_dir


def _rewrite_zip_manifest(zip_path: Path, manifest: dict[str, Any]) -> None:
    temp_path = zip_path.with_suffix(".tmp")
    with zipfile.ZipFile(zip_path) as archive, zipfile.ZipFile(
        temp_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as new_archive:
        for info in archive.infolist():
            if info.filename == "manifest.json":
                continue
            new_archive.writestr(info, archive.read(info.filename))
        new_archive.writestr(
            "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2)
        )
    zip_path.unlink()
    temp_path.rename(zip_path)


def format_schema_error(error: Any) -> str:
    path = "/".join(str(part) for part in error.path) or "<root>"
    message = str(getattr(error, "message", ""))
    max_len = 500
    if len(message) > max_len:
        message = f"{message[:max_len]}..."
    validator = getattr(error, "validator", None)
    schema_path = getattr(error, "schema_path", None)
    details: list[str] = []
    if validator:
        details.append(f"validator={validator}")
    if schema_path:
        details.append(f"schema_path={'/'.join(str(part) for part in schema_path)}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"{path}: {message}{suffix}"


def validate_manifest_schema(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if importlib.util.find_spec("jsonschema") is not None:
        try:
            if not SCHEMA_PATH.exists():
                return [f"schema file not found: {SCHEMA_PATH}"]
            schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
            jsonschema = importlib.import_module("jsonschema")
            validator_cls = jsonschema.validators.validator_for(schema)
            validator_cls.check_schema(schema)
            validator = validator_cls(schema)
            for error in sorted(
                validator.iter_errors(manifest), key=lambda err: list(err.path)
            ):
                errors.append(format_schema_error(error))
            return errors
        except Exception as exc:  # noqa: BLE001
            return [f"schema_validation_error: {exc.__class__.__name__}: {exc}"]

    print("Install agentforge[schema] for full schema validation.", file=sys.stderr)
    if manifest.get("spec_version") != SPEC_VERSION:
        errors.append("spec_version must be 0.1")
    for field in (
        "spec_version",
        "name",
        "version",
        "created_at",
        "publisher",
        "type",
        "files",
        "entrypoints",
    ):
        if field not in manifest:
            errors.append(f"missing required field: {field}")
    if manifest.get("type") and manifest.get("type") not in ALLOWED_TYPES:
        errors.append("type must be workflow_pack, tool_pack, or mixed")
    if "files" in manifest and not isinstance(manifest.get("files"), list):
        errors.append("files must be a list")
    if "entrypoints" in manifest and not isinstance(manifest.get("entrypoints"), list):
        errors.append("entrypoints must be a list")
    for entry in manifest.get("files", []):
        if not isinstance(entry, dict):
            errors.append("files entries must be objects")
            continue
        path = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(path, str):
            errors.append("file path must be a string")
        elif not _is_safe_relative_path(path):
            errors.append(f"file path is not safe: {path}")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append(f"invalid sha256 for {path}")
    return errors


def validate_pack(path: Path, key: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    schema_available = importlib.util.find_spec("jsonschema") is not None
    if path.is_dir():
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            return {"ok": False, "errors": ["manifest.json not found"], "warnings": []}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        errors.extend(validate_manifest_schema(manifest))
        for entry in manifest.get("files", []):
            file_path = path / entry["path"]
            if not file_path.exists():
                errors.append(f"missing file: {entry['path']}")
                continue
            if _hash_file(file_path) != entry["sha256"]:
                errors.append(f"hash mismatch for {entry['path']}")
    else:
        if not path.exists():
            return {"ok": False, "errors": ["pack not found"], "warnings": []}
        manifest = read_manifest_from_zip(path)
        errors.extend(validate_manifest_schema(manifest))
        if not verify_pack_files(path, manifest):
            errors.append("pack file hash verification failed")
    if not schema_available:
        warnings.append("Install agentforge[schema] for full schema validation.")
    signature = manifest.get("signature")
    if signature and key:
        if not verify_manifest(manifest, key):
            errors.append("signature verification failed")
    elif signature and not key:
        warnings.append("signature present but PACK_SIGNING_KEY not provided")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def safe_extract(archive: zipfile.ZipFile, dest: Path) -> None:
    dest_resolved = dest.resolve()
    for info in archive.infolist():
        path = info.filename
        if not _is_safe_relative_path(path):
            raise RuntimeError(f"Unsafe path in zip: {path}")
        normalized = Path(normpath(path))
        target_path = (dest / normalized).resolve()
        if dest_resolved not in target_path.parents and dest_resolved != target_path:
            raise RuntimeError(f"Path traversal blocked: {path}")
        if info.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target_path.open("wb") as target:
            target.write(source.read())


def _is_safe_relative_path(path: str) -> bool:
    if path.startswith(("/", "\\")) or ":" in path:
        return False
    parts = Path(path).parts
    if any(part == ".." for part in parts):
        return False
    return True
