"""Pack signing helpers."""

from __future__ import annotations

import base64
import hmac
from hashlib import sha256
import json
from typing import Any


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sign_manifest(manifest: dict[str, Any], key: str) -> str:
    payload = {k: v for k, v in manifest.items() if k != "signature"}
    raw = canonical_json(payload).encode("utf-8")
    digest = hmac.new(key.encode("utf-8"), raw, sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def verify_manifest(manifest: dict[str, Any], key: str) -> bool:
    signature = manifest.get("signature")
    if not signature:
        return False
    expected = sign_manifest(manifest, key)
    return hmac.compare_digest(signature, expected)
