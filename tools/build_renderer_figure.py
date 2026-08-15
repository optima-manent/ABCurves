"""Regenerate the README Renderer illustration from the final artifact."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from abcurves import Pipeline  # noqa: E402


def main() -> int:
    with np.load(ROOT / "examples" / "aim_test.npz", allow_pickle=False) as data:
        row = 0
        prefix = data["prefix_raw_dxdy"][row][data["prefix_mask"][row] > 0.5]
        context = np.zeros((256, 2), dtype=np.int16)
        context[-len(prefix) :] = np.rint(prefix).astype(np.int16)
        target = (
            float(data["target_rel_x_at_B"][row]),
            float(data["target_rel_y_at_B"][row]),
        )
        radius = float(data["target_radius"][row])
        progress = float(data["progress"][row])
        human = data["future_raw_dxdy"][row][data["future_mask"][row] > 0.5]

    with Pipeline.from_pretrained(prewarm=True) as pipeline:
        stream = pipeline.prepare(
            prefix,
            renderer_context_raw_dxdy=context,
            target_rel_at_B=target,
            target_radius=radius,
            progress_center=progress,
            planner_seed=2026,
            renderer_event_seed_u64=2026,
        )
        smooth = stream.planned.intent.smooth_dxdy[
            stream.planned.intent.mask > 0.5
        ]
        rendered = stream.render_remaining()

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4), dpi=180)
    palette = {"smooth": "#94a3b8", "rendered": "#7c3aed", "human": "#0f766e"}

    axes[0].plot(
        *np.cumsum(smooth, axis=0).T,
        color=palette["smooth"],
        linewidth=2.2,
        label="smooth plan",
    )
    axes[0].plot(
        *np.cumsum(rendered, axis=0).T,
        color=palette["rendered"],
        linewidth=1.2,
        label="rendered counts",
    )
    axes[0].set_title("Path preserved")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_xlabel("x counts")
    axes[0].set_ylabel("y counts")

    shown = min(140, len(rendered), len(smooth))
    ticks = np.arange(shown)
    axes[1].plot(
        ticks,
        smooth[:shown, 0],
        color=palette["smooth"],
        linewidth=2,
        label="smooth dx",
    )
    markerline, stems, _ = axes[1].stem(
        ticks, rendered[:shown, 0], linefmt="-", markerfmt=" ", basefmt=" "
    )
    plt.setp(stems, color=palette["rendered"], linewidth=0.8)
    plt.setp(markerline, color=palette["rendered"])
    axes[1].set_title("1 ms report texture")
    axes[1].set_xlabel("milliseconds")
    axes[1].set_ylabel("dx counts")

    max_mag = int(
        max(
            np.max(np.abs(rendered), initial=0),
            np.max(np.abs(human), initial=0),
            1,
        )
    )
    bins = np.arange(-0.5, max_mag + 1.5)
    axes[2].hist(
        np.max(np.abs(human), axis=1),
        bins=bins,
        density=True,
        alpha=0.6,
        color=palette["human"],
        label="real human reports",
    )
    axes[2].hist(
        np.max(np.abs(rendered), axis=1),
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        color=palette["rendered"],
        label="Renderer reports",
    )
    axes[2].set_title("Human-like packet sizes")
    axes[2].set_xlabel("max(|dx|, |dy|)")
    axes[2].set_ylabel("share of reports")
    axes[2].legend(frameon=False, fontsize=8)

    figure.suptitle(
        "The global Renderer turns a smooth plan into mouse reports",
        fontsize=11,
    )
    figure.tight_layout()
    output = ROOT / "assets" / "renderer_texture.png"
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
