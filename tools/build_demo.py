"""Regenerate the static demo payload with the final Planner and Renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from abcurves import Pipeline  # noqa: E402


PAYLOAD_PATTERN = re.compile(
    r'<script id="payload" type="application/json">(.*?)</script>', re.DOTALL
)


def _source_examples(path: Path) -> list[dict]:
    html = path.read_text(encoding="utf-8")
    match = PAYLOAD_PATTERN.search(html)
    if match is None:
        raise ValueError(f"{path}: demo payload is missing")
    value = json.loads(match.group(1))
    return list(value["examples"])


def _render_example(pipeline: Pipeline, source: dict, index: int) -> dict:
    prefix_path = np.asarray(source["prefix"], dtype=np.float64)
    if prefix_path.shape != (161, 2):
        raise ValueError(f"example {index}: expected 161 cumulative prefix points")
    prefix = np.diff(prefix_path, axis=0).astype(np.float32)
    if not np.array_equal(prefix, np.rint(prefix)):
        raise ValueError(f"example {index}: prefix is not physical integer counts")
    context = np.zeros((256, 2), dtype=np.int16)
    context[-len(prefix) :] = prefix.astype(np.int16)
    b = np.asarray(source["b"], dtype=np.float64)
    target = np.asarray(source["target"], dtype=np.float64)
    start = prefix_path[0]
    full_distance = float(np.linalg.norm(target - start))
    progress = 1.0 - float(np.linalg.norm(target - b)) / max(full_distance, 1e-9)
    progress = float(np.clip(progress, 0.0, 1.0))

    samples: list[list[list[float]]] = []
    for draw in range(4):
        seed = 20_000 + index * 16 + draw
        counts = pipeline.generate(
            prefix,
            renderer_context_raw_dxdy=context,
            target_rel_at_B=(float(target[0] - b[0]), float(target[1] - b[1])),
            target_radius=float(source["radius"]),
            progress_center=progress,
            seed=seed,
        )
        path = np.concatenate([b[None, :], b[None, :] + np.cumsum(counts, axis=0)])
        samples.append(path.astype(float).tolist())

    common = min(len(sample) for sample in samples)
    average = np.rint(
        np.mean([np.asarray(sample[:common]) for sample in samples], axis=0)
    ).astype(float)
    return {
        "kind": source["kind"],
        "user": source.get("user", "held-out example"),
        "prefix": prefix_path.astype(float).tolist(),
        "b": b.astype(float).tolist(),
        "target": target.astype(float).tolist(),
        "radius": float(source["radius"]),
        "real": source["real"],
        "samples": samples,
        "average": average.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "docs" / "index.html")
    parser.add_argument("--template", type=Path, default=ROOT / "docs" / "demo_template.html")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "index.html")
    args = parser.parse_args()
    source = _source_examples(args.source)
    with Pipeline.from_pretrained(prewarm=True) as pipeline:
        examples = [
            _render_example(pipeline, example, index)
            for index, example in enumerate(source)
        ]
        renderer_sha = pipeline.renderer_receipt["artifact_sha256"]
    payload = {
        "schema": "abcurves.demo.v1",
        "renderer_artifact_sha256": renderer_sha,
        "context_note": (
            "event fixtures contain 160 prefix reports; the preceding 96 reports "
            "are an explicit quiet-start assumption"
        ),
        "examples": examples,
    }
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    template = args.template.read_text(encoding="utf-8")
    if template.count("__PAYLOAD__") != 1:
        raise ValueError("demo template must contain exactly one __PAYLOAD__ marker")
    result = template.replace("__PAYLOAD__", encoded)
    args.out.write_text(result, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "examples": len(examples),
                "renderer_artifact_sha256": renderer_sha,
                "output_sha256": hashlib.sha256(result.encode("utf-8")).hexdigest(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
