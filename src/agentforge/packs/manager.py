"""Pack build/sign/install helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import zipfile
from typing import Any

from agentforge.packs.signing import sign_manifest, verify_manifest


@dataclass
class PackManifest:
    name: str
    version: str
    pack_type: str
    created_at: str
    publisher: str
    entrypoints: dict[str, Any]
    files: list[dict[str, str]]
    signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
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
        entrypoints = payload.get("entrypoints") or {}
    else:
        name = source_dir.name
        version = "0.0.0"
        pack_type = "mixed"
        publisher = "unknown"
        entrypoints = {}
    files: list[dict[str, str]] = []
    for path in source_dir.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(source_dir).as_posix()
        if relative == "manifest.json":
            continue
        files.append({"path": relative, "sha256": _hash_file(path)})
    return PackManifest(
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


def sign_pack(zip_path: Path, key: str) -> dict[str, Any]:
    manifest = read_manifest_from_zip(zip_path)
    manifest["signature"] = sign_manifest(manifest, key)
    _rewrite_zip_manifest(zip_path, manifest)
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
        archive.extractall(pack_dir)
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
