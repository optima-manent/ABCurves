"""Train the Renderer (count-texture model) and export a checkpoint.

This is the recipe behind the shipped Renderer. The Renderer learns hardware
texture from exactly one
successful B80 continuation per source movement.  From each retained stream we
make a ``(smooth(raw), raw)`` pair and teach the model, tick by tick, to turn
the smooth version back into integer hardware counts.

What it learns per tick, given causal features (smooth kinematics, accumulator
state, previous emission, run state, recent zero rate, prefix count-regime):

* ``emit`` -- report a packet now? (reproduces the bursty zero/nonzero cadence)
* ``offset`` -- a small integer deviation from the delta-sigma accumulator's
  rounded base (reproduces the packet magnitude/direction texture)

The accumulator reclaims whatever it emits, so the endpoint never drifts.

Usage:

    python training/train_renderer.py \
        --train prepared/renderer_train.npz \
        --out runs/renderer_retrained.pt --epochs 12

Use the Renderer branch emitted by ``tools/prepare_dataset.py``. It contains
exactly one successful B80 row per retained physical source.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abcurves.data import load_dataset
from abcurves.preprocessing import PREPARED_CUT_SCHEMA
from abcurves.renderer import (
    RendererConfig,
    benchmark_tick_latency,
    save_count_model,
    train_count_texture_model,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_text(arrays: dict, key: str) -> str | None:
    if key not in arrays:
        return None
    values = arrays[key].reshape(-1)
    if values.size != 1:
        raise ValueError(f"{key} must contain exactly one dataset-level value")
    value = values[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def require_renderer_dataset(arrays: dict) -> None:
    """Require the fixed-B80 Renderer branch from the release builder."""

    schema = _metadata_text(arrays, "schema")
    cohort = _metadata_text(arrays, "cohort")
    if schema != PREPARED_CUT_SCHEMA or not str(cohort).startswith("renderer_"):
        raise ValueError(
            "--train must be a Renderer split emitted by tools/prepare_dataset.py"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--train",
        required=True,
        help="prepared renderer_train.npz emitted by tools/prepare_dataset.py",
    )
    ap.add_argument("--out", default="runs/renderer_retrained.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--offset-radius", type=int, default=5, help="max integer offset the head can emit")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # The final Renderer population is successful terminal aim movement at B80.
    # Planner-only geometry/tiny-target filters do not apply to this branch.
    train_path = Path(args.train)
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing renderer: {out_path}")
    arrays = load_dataset(train_path, success_only=True)
    require_renderer_dataset(arrays)
    print(f"train {len(arrays['future_mask'])} raw streams, device {args.device}")

    cfg = RendererConfig(
        offset_radius=args.offset_radius,
        epochs=args.epochs,
        seed=args.seed,
    )
    t0 = time.time()
    model, report = train_count_texture_model(arrays, cfg, device=args.device, log_every=5)
    report["train_npz"] = str(train_path.resolve())
    report["train_npz_sha256"] = _sha256(train_path)
    report["train_schema"] = (
        str(arrays["schema"].reshape(-1)[0]) if "schema" in arrays else None
    )
    report["train_cohort"] = (
        str(arrays["cohort"].reshape(-1)[0]) if "cohort" in arrays else None
    )
    save_count_model(model, report, out_path)

    lat = benchmark_tick_latency(model)
    print(
        f"wrote {args.out}  ({report['n_parameters']} params, "
        f"clip_fraction {report['teacher_offset_clip_fraction']:.4f}, {time.time()-t0:.0f}s)"
    )
    print(f"per-tick CPU latency {lat['per_tick_us']:.1f} us  (1 kHz budget = 1000 us/tick)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
