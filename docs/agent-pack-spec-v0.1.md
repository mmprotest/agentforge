# Agent Pack Specification v0.1

## 1. Goals

* Provide a stable, portable format for distributing AgentForge workflow and tool packs.
* Define deterministic hashing and signing rules for integrity and authenticity.
* Enable secure installation with mandatory path traversal protection.

## 2. Non-Goals

* Defining runtime semantics of workflows or tools.
* Defining public-key signature formats (reserved for future profiles).
* Hosting or distribution mechanisms.

## 3. Terminology

* **Pack**: A directory or zip archive containing AgentForge assets plus a manifest.
* **Manifest**: `manifest.json` describing pack metadata, file hashes, and entrypoints.
* **Entrypoint**: A workflow id or tool name to expose after install.
* **Signature**: An optional proof over a canonicalized manifest.

## 4. Pack Structure

### 4.1 Directory

```
pack/
  manifest.json
  README.md
  workflows/...
  tools/...
```

### 4.2 Zip

* A zip archive with `manifest.json` at the root.
* All paths are relative and use `/` separators.

## 5. Manifest (`manifest.json`)

### 5.1 Required fields

* `spec_version`: `"0.1"`
* `name`: string
* `version`: string
* `created_at`: string (ISO-8601 recommended)
* `publisher`: string
* `type`: `"workflow_pack"` | `"tool_pack"` | `"mixed"`
* `files`: list of `{ "path": string, "sha256": string }`
* `entrypoints`: list of workflow ids and/or tool names

### 5.2 Optional fields

* `signature`: string

## 6. Canonical JSON for Signing

* UTF-8 encoded.
* Sorted keys.
* No insignificant whitespace.
* The signature is computed over the manifest **without** the `signature` field.

## 7. Hashing Rules

* SHA-256 over the raw bytes of each file.
* `files[].path` uses forward slashes (`/`) regardless of OS.

## 8. Signing Profiles

### Profile A (HMAC-SHA256, default)

```
signature = base64(hmac_sha256(key, canonical_manifest_without_signature))
```

* `key` is provided out-of-band (env var or secret reference).

### Profile B (reserved)

* Reserved for public-key signing (not implemented in v0.1).

## 9. Installation Security Requirements

* MUST prevent Zip Slip (path traversal).
* MUST verify file hashes before activation.
* SHOULD verify signatures when present.

## 10. Versioning and Compatibility

* `spec_version` is required and currently `"0.1"`.
* Clients MUST reject unknown `spec_version` values.
* Minor revisions may add optional fields without breaking compatibility.
