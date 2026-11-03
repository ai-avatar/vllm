# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin inductor AutotuneCache ``*.best_config`` files per GPU SKU.

Multi-config inductor kernels (e.g. rms_norm R0_BLOCK 4096 vs 2048) are
timed once at cold compile. The winner is stored as a ``.best_config``
file under ``TORCHINDUCTOR_CACHE_DIR``. Loading those files *before*
``CompiledArtifact.load`` / ``CachingAutotuner.precompile`` makes every
autotuner a cache hit (or already single-config), so warmup never
re-times.

Pinned choices are GPU- and torch-version-specific. Generate a pack on
the target SKU (A100 sm80 vs L40S sm89) and keep it keyed by device
family plus compute capability. See
``docs/features/batch_invariance.md`` for the pod regen steps.

    python -m vllm.compilation.inductor_autotune_pin sku
    python -m vllm.compilation.inductor_autotune_pin dump \\
        --cache-dir $TORCHINDUCTOR_CACHE_DIR --output /path/to/pack
    python -m vllm.compilation.inductor_autotune_pin install \\
        --source /path/to/pack --cache-dir $TORCHINDUCTOR_CACHE_DIR
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import regex as re

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1
BEST_CONFIG_SUFFIX = ".best_config"

_SKU_ALIASES: tuple[tuple[str, str], ...] = (
    ("a100", "a100"),
    ("l40s", "l40s"),
    ("l40", "l40"),
    ("h100", "h100"),
    ("h200", "h200"),
    ("b200", "b200"),
)


def current_sku_key(
    *,
    gpu_name: str | None = None,
    compute_capability: tuple[int, int] | None = None,
) -> str:
    """Return a stable pack key such as ``a100-sm80`` or ``l40s-sm89``."""
    name = gpu_name
    cap = compute_capability
    if name is None or cap is None:
        try:
            from vllm.platforms import current_platform

            if name is None:
                name = current_platform.get_device_name()
            if cap is None:
                device_cap = current_platform.get_device_capability()
                if device_cap is not None:
                    cap = (int(device_cap.major), int(device_cap.minor))
                elif current_platform.is_cpu():
                    return "cpu"
        except Exception:  # noqa: BLE001
            if name is None and cap is None:
                return "unknown"
    family = "gpu"
    lowered = (name or "unknown").lower()
    for needle, alias in _SKU_ALIASES:
        if needle in lowered:
            family = alias
            break
    else:
        family = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-") or "gpu"
        family = family.removeprefix("nvidia-")[:48]
    sm = "unknown" if cap is None else f"sm{cap[0]}{cap[1]}"
    return f"{family}-{sm}"


def packaged_pin_root() -> Path:
    return Path(__file__).resolve().parent / "inductor_autotune_pins"


def default_cache_pin_dir(sku_key: str | None = None) -> Path:
    sku = sku_key or current_sku_key()
    return Path(envs.VLLM_CACHE_ROOT) / "inductor_autotune_pins" / sku


def _pack_has_configs(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.rglob(f"*{BEST_CONFIG_SUFFIX}"))


def resolve_pin_pack_dir(sku_key: str | None = None) -> Path | None:
    """Locate a pin pack: env override, then packaged, then cache dump."""
    override = envs.VLLM_INDUCTOR_AUTOTUNE_PIN_DIR
    if override:
        path = Path(override).expanduser()
        return path if path.is_dir() else None
    sku = sku_key or current_sku_key()
    packaged = packaged_pin_root() / sku
    if _pack_has_configs(packaged):
        return packaged
    cached = default_cache_pin_dir(sku)
    if _pack_has_configs(cached):
        return cached
    return None


def inductor_cache_dir() -> Path | None:
    raw = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_dir() else None


def iter_best_config_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob(f"*{BEST_CONFIG_SUFFIX}") if p.is_file())


def _safe_relpath(path: Path, root: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if not parts or any(p in (os.pardir, "") for p in parts):
        return None
    return rel.as_posix()


def read_manifest(pack_dir: Path) -> dict[str, Any] | None:
    manifest_path = pack_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    with manifest_path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    return data


def write_manifest(pack_dir: Path, payload: dict[str, Any]) -> None:
    pack_dir.mkdir(parents=True, exist_ok=True)
    tmp = pack_dir / f".{MANIFEST_NAME}.tmp"
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(pack_dir / MANIFEST_NAME)


def _torch_triton_versions() -> tuple[str, str | None]:
    torch_version = "unknown"
    triton_version = None
    try:
        import torch

        torch_version = str(torch.__version__)
    except Exception:  # noqa: BLE001
        pass
    try:
        import triton

        triton_version = str(getattr(triton, "__version__", None))
    except Exception:  # noqa: BLE001
        pass
    return torch_version, triton_version


def dump_pinned_autotune_cache(
    cache_dir: Path | str,
    output_dir: Path | str,
    *,
    sku_key: str | None = None,
    gpu_name: str | None = None,
) -> dict[str, Any]:
    """Copy ``*.best_config`` files from an inductor cache into a pin pack."""
    src = Path(cache_dir)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for path in iter_best_config_files(src):
        rel = _safe_relpath(path, src)
        if rel is None:
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        files.append(rel)
    sku = sku_key or current_sku_key(gpu_name=gpu_name)
    torch_version, triton_version = _torch_triton_versions()
    cap = None
    try:
        from vllm.platforms import current_platform

        device_cap = current_platform.get_device_capability()
        if device_cap is not None:
            cap = f"{device_cap.major}.{device_cap.minor}"
        if gpu_name is None:
            gpu_name = current_platform.get_device_name()
    except Exception:  # noqa: BLE001
        pass
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sku_key": sku,
        "gpu_name": gpu_name or "",
        "compute_capability": cap or "",
        "torch_version": torch_version,
        "triton_version": triton_version,
        "created_at": datetime.now(UTC).isoformat(),
        "file_count": len(files),
        "files": files,
    }
    write_manifest(dest, manifest)
    logger.info(
        "Dumped %d inductor autotune .best_config files to %s (sku=%s)",
        len(files),
        dest,
        sku,
    )
    return manifest


def install_pinned_autotune_cache(
    cache_dir: Path | str,
    source_dir: Path | str | None = None,
    *,
    sku_key: str | None = None,
) -> int:
    """Copy a pin pack into ``TORCHINDUCTOR_CACHE_DIR`` before precompile.

    Returns:
        Number of ``.best_config`` files installed. 0 if no pack is found.

    """
    dest = Path(cache_dir)
    dest.mkdir(parents=True, exist_ok=True)
    pack = Path(source_dir) if source_dir is not None else resolve_pin_pack_dir(sku_key)
    if pack is None:
        logger.info(
            "No pinned inductor autotune pack for sku=%s "
            "(set VLLM_INDUCTOR_AUTOTUNE_PIN_DIR or dump one after a "
            "reference cold compile)",
            sku_key or current_sku_key(),
        )
        return 0
    manifest = read_manifest(pack)
    if manifest:
        torch_version, _ = _torch_triton_versions()
        pinned_torch = manifest.get("torch_version")
        if pinned_torch and torch_version and pinned_torch != torch_version:
            logger.warning(
                "Pinned inductor autotune pack torch_version=%s does not "
                "match this process (%s); keys may miss and warmup will "
                "re-benchmark",
                pinned_torch,
                torch_version,
            )
        pinned_sku = manifest.get("sku_key")
        running_sku = sku_key or current_sku_key()
        if pinned_sku and pinned_sku != running_sku:
            logger.warning(
                "Pinned inductor autotune pack sku=%s does not match this device (%s)",
                pinned_sku,
                running_sku,
            )
    installed = 0
    for path in iter_best_config_files(pack):
        rel = _safe_relpath(path, pack)
        if rel is None:
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        installed += 1
    logger.info(
        "Installed %d pinned inductor autotune configs from %s into %s",
        installed,
        pack,
        dest,
    )
    return installed


def pin_dump_dir() -> Path | None:
    """Return the dump destination when ``VLLM_INDUCTOR_AUTOTUNE_PIN_DUMP`` is set."""
    raw = envs.VLLM_INDUCTOR_AUTOTUNE_PIN_DUMP
    if not raw:
        return None
    value = raw.strip()
    if value.lower() in ("0", "false", "no", "off"):
        return None
    if value.lower() in ("1", "true", "yes", "on"):
        override = envs.VLLM_INDUCTOR_AUTOTUNE_PIN_DIR
        if override:
            return Path(override).expanduser()
        return default_cache_pin_dir()
    return Path(value).expanduser()


def maybe_dump_pinned_autotune_cache(
    cache_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    dest = pin_dump_dir()
    if dest is None:
        return None
    src = Path(cache_dir) if cache_dir is not None else inductor_cache_dir()
    if src is None:
        logger.warning(
            "VLLM_INDUCTOR_AUTOTUNE_PIN_DUMP is set but TORCHINDUCTOR_CACHE_DIR "
            "is missing; skip dump"
        )
        return None
    try:
        return dump_pinned_autotune_cache(src, dest)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to dump inductor autotune pin pack to %s: %s", dest, exc)
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dump or install pinned inductor AutotuneCache results.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sku", help="Print the current GPU SKU key")
    dump = sub.add_parser("dump", help="Copy *.best_config files into a pin pack")
    dump.add_argument(
        "--cache-dir",
        default=os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        help="Inductor cache directory (default: TORCHINDUCTOR_CACHE_DIR)",
    )
    dump.add_argument(
        "--output",
        default=None,
        help="Pin pack directory (default: cache_root/inductor_autotune_pins/<sku>)",
    )
    install = sub.add_parser(
        "install", help="Copy a pin pack into TORCHINDUCTOR_CACHE_DIR"
    )
    install.add_argument(
        "--source",
        default=None,
        help="Pin pack directory (default: resolve for this SKU)",
    )
    install.add_argument(
        "--cache-dir",
        default=os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        help="Destination inductor cache directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "sku":
        print(current_sku_key())
        return 0
    if args.command == "dump":
        if not args.cache_dir:
            parser.error("dump requires --cache-dir or TORCHINDUCTOR_CACHE_DIR")
        output = Path(args.output) if args.output else default_cache_pin_dir()
        dump_pinned_autotune_cache(args.cache_dir, output)
        return 0
    if args.command == "install":
        if not args.cache_dir:
            parser.error("install requires --cache-dir or TORCHINDUCTOR_CACHE_DIR")
        n = install_pinned_autotune_cache(args.cache_dir, args.source)
        if n == 0:
            return 1
        return 0
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
