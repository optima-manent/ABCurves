"""Render the compact warm-versus-cold release figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(results: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    warm = json.loads((results / "warm_known_reference.json").read_text(encoding="utf-8"))
    cold = json.loads((results / "cold_unknown_person.json").read_text(encoding="utf-8"))
    warm_rows = warm["mixture_power"]
    cold_rows = cold["complete_detector_power"]
    warm_by_count = {int(row["generated_rows"]): float(row["flag_rate"]) for row in warm_rows}
    cold_by_count = {
        int(row["generated_rows"]): float(row["leave_key_out_flag_rate"])
        for row in cold_rows
    }
    counts = sorted(set(warm_by_count) & set(cold_by_count))
    x = [100.0 * count / 32.0 for count in counts]
    figure, axis = plt.subplots(figsize=(10.8, 6.4), constrained_layout=True)
    axis.plot(
        x,
        [warm_by_count[count] for count in counts],
        marker="o",
        linewidth=2.8,
        color="#2563eb",
        label="Warm: trusted same-session reference",
    )
    axis.plot(
        x,
        [cold_by_count[count] for count in counts],
        marker="o",
        linewidth=2.8,
        color="#dc2626",
        label="Cold: previously unseen installation key",
    )
    axis.set(
        title="Detection depends on what the detector already knows",
        xlabel="Generated rows in a 32-curve bag (%)",
        ylabel="Observed flag rate",
        xlim=(-2, 102),
        ylim=(-0.035, 1.035),
    )
    axis.grid(alpha=0.22)
    axis.legend(loc="upper left", frameon=True)
    axis.text(
        98,
        0.14,
        "Cold study: 1 of 6 held keys\nwas falsely flagged; safe components\nhad zero E260 power.",
        ha="right",
        va="bottom",
        fontsize=10,
        color="#7f1d1d",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fef2f2", "edgecolor": "#fecaca"},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=root / "results" / "detection")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "detection" / "detection_modes.png",
    )
    arguments = parser.parse_args()
    render(arguments.results, arguments.output)


if __name__ == "__main__":
    main()
