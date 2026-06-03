#!/usr/bin/env python3
"""Convenience wrapper for converting OpenPI JAX checkpoints to PyTorch format.

This wrapper keeps the canonical converter in `examples/convert_jax_model_to_pytorch.py`
as the single source of truth, while providing a stable script entrypoint under `scripts/`.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, help="Path to JAX checkpoint directory (e.g. .../2500)")
    parser.add_argument("--config-name", required=True, help="Train config name (e.g. pi05_garo)")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output directory for converted PyTorch checkpoint. Defaults to <checkpoint-dir>_pytorch.",
    )
    parser.add_argument(
        "--precision",
        default="bfloat16",
        choices=("float32", "bfloat16", "float16"),
        help="Conversion precision.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect JAX checkpoint keys only (do not convert).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    converter = repo_root / "examples" / "convert_jax_model_to_pytorch.py"

    checkpoint_dir = pathlib.Path(args.checkpoint_dir).resolve()
    output_path = args.output_path
    if output_path is None and not args.inspect_only:
        output_path = str(checkpoint_dir) + "_pytorch"

    cmd = [
        sys.executable,
        str(converter),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--config-name",
        args.config_name,
        "--precision",
        args.precision,
    ]

    if args.inspect_only:
        cmd.append("--inspect-only")
    else:
        cmd.extend(["--output-path", output_path])

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not args.inspect_only:
        print(f"Converted checkpoint saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
