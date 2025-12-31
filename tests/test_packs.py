from pathlib import Path
import json
import zipfile

from agentforge.packs.manager import build_pack, install_pack, sign_pack, verify_pack


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
        json.dumps({"name": "demo", "version": "1.0.0", "type": "tool_pack"}),
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
        json.dumps({"name": "demo", "version": "1.0.0", "type": "tool_pack"}),
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
