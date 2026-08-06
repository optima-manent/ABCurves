"""Read and profile ``abcurves.native_count_1khz.v1`` export directories.

This is the auditable bridge between the capture trainer's transparent
interchange and ABCurves model datasets.  It deliberately produces event-level
measurements first.  Cohort selection and B augmentation happen only after the
measurements can be inspected per user and session.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .capture_preprocess import (
    CausalBConfig,
    CausalOnsetConfig,
    OnsetConfig,
    SeamEligibility,
    ShotFilterPolicy,
    causal_seam_contract_record,
    dense_slice_indices,
    detect_movement_onset,
    edge_progress_decision,
    post_c_tail_metrics,
    shot_filter_reasons,
    shot_quality,
)


EXPORT_SCHEMA = "abcurves.research_export.v1"
ADAPTER_ID = "abcurves.native_count_1khz.v1"
PROFILE_SCHEMA = "abcurves.capture_profile.v3"

MOUSE_COLUMNS = (
    "begin_unix_ns",
    "end_unix_ns",
    "canonical_dx",
    "canonical_dy",
    "buttons_down_mask",
    "report_count",
    "zero_delta_report_count",
    "quality_mask",
    "button_edges_json",
)

EVENT_COLUMNS = (
    "session_id",
    "user_id",
    "event_id",
    "block_ordinal",
    "target_ordinal_in_block",
    "challenge_id",
    "task_type",
    "target_role",
    "trainer_sensitivity",
    "first_presented_unix_ns",
    "event_start_unix_ns",
    "natural_resolution_unix_ns",
    "technical_interruption_unix_ns",
    "tail_end_unix_ns",
    "target_x_counts",
    "target_y_counts",
    "target_radius_counts",
    "relative_target_x_counts",
    "relative_target_y_counts",
    "initial_distance_counts",
    "start_crosshair_x_counts",
    "start_crosshair_y_counts",
    "generation_camera_x_counts",
    "generation_camera_y_counts",
    "presentation_camera_x_counts",
    "presentation_camera_y_counts",
    "natural_outcome",
    "technical_outcome",
    "inside_total_ms",
    "maximum_consecutive_inside_ms",
    "observed_tail_ms",
    "click_count",
    "click_hypotheses_json",
    "clock_fit_warning_mask",
)


def _required_columns(frame: pd.DataFrame, names: Iterable[str], *, label: str) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hash(manifest: dict[str, Any], relative_path: str) -> str:
    matches = [
        str(item.get("sha256", ""))
        for item in manifest.get("artifacts", [])
        if item.get("relative_path") == relative_path
    ]
    if len(matches) != 1 or len(matches[0]) != 64:
        raise ValueError(f"export manifest lacks one valid hash for {relative_path}")
    return matches[0]


def load_export_manifest(export_dir: str | Path) -> dict[str, Any]:
    root = Path(export_dir)
    manifest = json.loads((root / "export_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != EXPORT_SCHEMA:
        raise ValueError(f"unsupported export schema: {manifest.get('schema')!r}")
    if manifest.get("adapter_id") != ADAPTER_ID:
        raise ValueError(f"unsupported adapter: {manifest.get('adapter_id')!r}")
    if bool(manifest.get("continuation", {}).get("b_seam_defined")):
        raise ValueError("capture interchange unexpectedly defines a B seam")
    return manifest


def load_export_tables(export_dir: str | Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Load one validated research export."""

    root = Path(export_dir)
    manifest = load_export_manifest(root)
    # Nullable Unix-nanosecond columns must not round-trip through float64.
    # Current values are around 1.8e18, where an IEEE-754 float cannot retain
    # the exact integer and can move a boundary into a neighbouring 1 ms bin.
    events = pd.read_csv(
        root / "trainer_events.csv",
        usecols=list(EVENT_COLUMNS),
        dtype={
            "first_presented_unix_ns": "Int64",
            "event_start_unix_ns": "Int64",
            "natural_resolution_unix_ns": "Int64",
            "technical_interruption_unix_ns": "Int64",
            "tail_end_unix_ns": "Int64",
        },
    )
    mouse = pd.read_csv(root / "mouse_1ms.csv", usecols=list(MOUSE_COLUMNS))
    _required_columns(events, EVENT_COLUMNS, label="trainer_events.csv")
    _required_columns(mouse, MOUSE_COLUMNS, label="mouse_1ms.csv")

    begin = mouse["begin_unix_ns"].to_numpy(np.int64)
    end = mouse["end_unix_ns"].to_numpy(np.int64)
    if len(begin):
        if np.any(end <= begin) or np.any(begin[1:] != end[:-1]):
            raise ValueError("mouse_1ms.csv is not a consecutive half-open dense grid")
        expected = int(manifest["dense_grid"]["period_ns"])
        if np.any(end - begin != expected):
            raise ValueError("mouse_1ms.csv does not use the declared dense period")
    if int(manifest["counts"]["trainer_events"]) != len(events):
        raise ValueError("trainer event count disagrees with export manifest")
    if int(manifest["counts"]["dense_millisecond_bins"]) != len(mouse):
        raise ValueError("dense-bin count disagrees with export manifest")
    return manifest, events, mouse


def nearest_usb_button_down(
    begin_unix_ns: np.ndarray,
    end_unix_ns: np.ndarray,
    buttons_down_mask: np.ndarray,
    reference_unix_ns: int,
    *,
    button_mask: int = 1,
    before_ms: float = 30.0,
    after_ms: float = 10.0,
) -> tuple[int, float] | None:
    """Find the nearest authoritative USB button-down bin.

    Returns ``(row_index, midpoint_minus_reference_ms)``.  The asymmetric
    search window reflects that USB capture normally precedes Windows Raw Input
    receipt, but a small positive tolerance is retained for clock/join jitter.
    """

    begin = np.asarray(begin_unix_ns, dtype=np.int64)
    end = np.asarray(end_unix_ns, dtype=np.int64)
    buttons = np.asarray(buttons_down_mask, dtype=np.int64)
    if not (len(begin) == len(end) == len(buttons)):
        raise ValueError("button lookup arrays must have equal length")
    ref = int(reference_unix_ns)
    lo_ns = ref - int(round(float(before_ms) * 1_000_000.0))
    hi_ns = ref + int(round(float(after_ms) * 1_000_000.0))
    lo = int(np.searchsorted(begin, lo_ns, side="left"))
    hi = int(np.searchsorted(begin, hi_ns, side="right"))
    candidates = np.flatnonzero((buttons[lo:hi] & int(button_mask)) != 0) + lo
    if len(candidates) == 0:
        return None
    midpoint = begin[candidates] + (end[candidates] - begin[candidates]) // 2
    chosen_at = int(np.argmin(np.abs(midpoint - ref)))
    chosen = int(candidates[chosen_at])
    offset_ms = float(midpoint[chosen_at] - ref) / 1_000_000.0
    return chosen, offset_ms


def exact_usb_button_down_edges(
    begin_unix_ns: np.ndarray,
    end_unix_ns: np.ndarray,
    button_edges_json: Iterable[Any],
    *,
    button_mask: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract exact authoritative button-down timestamps and dense row IDs."""

    begin = np.asarray(begin_unix_ns, dtype=np.int64)
    end = np.asarray(end_unix_ns, dtype=np.int64)
    encoded = list(button_edges_json)
    if not (len(begin) == len(end) == len(encoded)):
        raise ValueError("button-edge arrays must have equal length")
    times: list[int] = []
    rows: list[int] = []
    previous_time: int | None = None
    for row_index, raw in enumerate(encoded):
        if not isinstance(raw, str) or raw in {"", "[]"}:
            continue
        try:
            edges = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid button_edges_json at dense row {row_index}") from error
        if not isinstance(edges, list):
            raise ValueError(f"button_edges_json must be a list at dense row {row_index}")
        for edge in edges:
            try:
                down_mask = int(edge["down_mask"])
                capture_ns = int(edge["capture_unix_ns"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid button edge at dense row {row_index}"
                ) from error
            if not (int(begin[row_index]) <= capture_ns < int(end[row_index])):
                raise ValueError(
                    f"button edge timestamp lies outside dense row {row_index}"
                )
            if previous_time is not None and capture_ns < previous_time:
                raise ValueError("button edges are not time ordered")
            previous_time = capture_ns
            if down_mask & int(button_mask):
                times.append(capture_ns)
                rows.append(row_index)
    return np.asarray(times, dtype=np.int64), np.asarray(rows, dtype=np.int64)


def monotone_button_matches(
    reference_unix_ns: Iterable[int | None],
    edge_unix_ns: np.ndarray,
    *,
    before_ms: float = 30.0,
    after_ms: float = 10.0,
) -> np.ndarray:
    """Greedily match ordered resolutions to unique, ordered USB edges.

    The nearest admissible edge is selected, but an edge can resolve at most
    one event and later events can never reuse an earlier edge.
    """

    refs = list(reference_unix_ns)
    edges = np.asarray(edge_unix_ns, dtype=np.int64)
    if len(edges) and np.any(edges[1:] < edges[:-1]):
        raise ValueError("edge_unix_ns must be sorted")
    valid_refs = [
        (index, int(value))
        for index, value in enumerate(refs)
        if value is not None
    ]
    if any(valid_refs[i][1] > valid_refs[i + 1][1] for i in range(len(valid_refs) - 1)):
        raise ValueError("non-null reference timestamps must be sorted")
    out = np.full(len(refs), -1, dtype=np.int64)
    minimum_edge = 0
    before_ns = int(round(float(before_ms) * 1_000_000.0))
    after_ns = int(round(float(after_ms) * 1_000_000.0))
    for output_index, reference in valid_refs:
        lo = max(
            minimum_edge,
            int(np.searchsorted(edges, reference - before_ns, side="left")),
        )
        hi = int(np.searchsorted(edges, reference + after_ns, side="right"))
        if lo >= hi:
            continue
        candidates = np.arange(lo, hi, dtype=np.int64)
        chosen = int(candidates[np.argmin(np.abs(edges[candidates] - reference))])
        out[output_index] = chosen
        minimum_edge = chosen + 1
    return out


def _presentation_target(row: Any) -> np.ndarray:
    """Target-center vector from the cursor camera at first presentation.

    ``relative_target_*`` is generation-relative, whereas
    ``initial_distance_counts`` and the exported dense event start are
    presentation-relative.  The cursor may move between generation and first
    presentation, so mixing those origins biases every downstream endpoint and
    progress measurement.
    """

    return np.asarray(
        [
            float(row.target_x_counts) - float(row.presentation_camera_x_counts),
            float(row.target_y_counts) - float(row.presentation_camera_y_counts),
        ],
        dtype=np.float64,
    )


def _generation_target(row: Any) -> np.ndarray:
    return np.asarray(
        [
            float(row.target_x_counts) - float(row.generation_camera_x_counts),
            float(row.target_y_counts) - float(row.generation_camera_y_counts),
        ],
        dtype=np.float64,
    )


def _raw_input_click_endpoint(row: Any) -> np.ndarray | None:
    raw = getattr(row, "click_hypotheses_json", None)
    if not isinstance(raw, str) or raw in {"", "[]"}:
        return None
    try:
        clicks = json.loads(raw)
    except json.JSONDecodeError:
        return None
    resolved = [item for item in clicks if bool(item.get("resolved_event"))]
    if not resolved:
        return None
    click = resolved[0]
    try:
        return np.asarray(
            [
                float(click["post_delta_x_counts"])
                - float(row.presentation_camera_x_counts),
                float(click["post_delta_y_counts"])
                - float(row.presentation_camera_y_counts),
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None


def profile_export_session(
    export_dir: str | Path,
    *,
    filter_policy: ShotFilterPolicy | None = None,
    progress_thresholds: tuple[float, ...] = (0.70, 0.80, 0.90),
    onset_config: OnsetConfig | None = None,
    seam_eligibility: SeamEligibility | None = None,
    slice_policy: str = "fully_contained",
    prefer_usb_click: bool = True,
    usb_click_button_mask: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Profile all events in one export and return rows plus a summary."""

    manifest, events, mouse = load_export_tables(export_dir)
    export_root = Path(export_dir)
    export_manifest_sha256 = _sha256_file(export_root / "export_manifest.json")
    mouse_sha256 = _artifact_hash(manifest, "mouse_1ms.csv")
    trainer_events_sha256 = _artifact_hash(manifest, "trainer_events.csv")
    policy = filter_policy or ShotFilterPolicy()
    onset_policy = onset_config or CausalOnsetConfig()
    eligibility = seam_eligibility or SeamEligibility()
    b_configs = tuple(CausalBConfig(float(value)) for value in progress_thresholds)
    contract_record = causal_seam_contract_record(
        onset_policy,
        b_configs,
        eligibility,
    )
    contract_json = json.dumps(
        contract_record,
        sort_keys=True,
        separators=(",", ":"),
    )
    begin = mouse["begin_unix_ns"].to_numpy(np.int64)
    end = mouse["end_unix_ns"].to_numpy(np.int64)
    raw_all = mouse[["canonical_dx", "canonical_dy"]].to_numpy(np.float64)
    report_count = mouse["report_count"].to_numpy(np.int64)
    zero_reports = mouse["zero_delta_report_count"].to_numpy(np.int64)
    quality_mask = mouse["quality_mask"].to_numpy(np.int64)
    usb_edge_times, usb_edge_rows = exact_usb_button_down_edges(
        begin,
        end,
        mouse["button_edges_json"].tolist(),
        button_mask=usb_click_button_mask,
    )
    click_references: list[int | None] = []
    for event in events.itertuples(index=False):
        is_click = str(event.natural_outcome) in {"hit_click", "miss_click"}
        click_references.append(
            int(event.natural_resolution_unix_ns)
            if is_click and pd.notna(event.natural_resolution_unix_ns)
            else None
        )
    matched_edge_positions = (
        monotone_button_matches(click_references, usb_edge_times)
        if prefer_usb_click
        else np.full(len(events), -1, dtype=np.int64)
    )

    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for event_row_index, event in enumerate(events.itertuples(index=False)):
        source_trial_id = f"{event.session_id}:{event.event_id}"
        presentation_target = _presentation_target(event)
        generation_target = _generation_target(event)
        event_start_ns = (
            int(event.event_start_unix_ns)
            if pd.notna(event.event_start_unix_ns)
            else None
        )
        resolution_ns_optional = (
            int(event.natural_resolution_unix_ns)
            if pd.notna(event.natural_resolution_unix_ns)
            else None
        )
        first_presented_ns = (
            int(event.first_presented_unix_ns)
            if pd.notna(event.first_presented_unix_ns)
            else None
        )
        tail_end_ns = (
            int(event.tail_end_unix_ns)
            if pd.notna(event.tail_end_unix_ns)
            else None
        )
        presentation_shift = np.asarray(
            [
                float(event.presentation_camera_x_counts)
                - float(event.generation_camera_x_counts),
                float(event.presentation_camera_y_counts)
                - float(event.generation_camera_y_counts),
            ],
            dtype=np.float64,
        )
        base: dict[str, Any] = {
            "source_trial_id": source_trial_id,
            "causal_seam_contract_schema": contract_record["schema"],
            "causal_seam_contract_json": contract_json,
            "source_export_manifest_sha256": export_manifest_sha256,
            "source_mouse_1ms_sha256": mouse_sha256,
            "source_trainer_events_sha256": trainer_events_sha256,
            "user_id": str(event.user_id),
            "session_id": str(event.session_id),
            "event_id": str(event.event_id),
            "block_ordinal": int(event.block_ordinal),
            "target_ordinal_in_block": int(event.target_ordinal_in_block),
            "challenge_id": str(event.challenge_id),
            "task_type": str(event.task_type),
            "target_role": str(event.target_role),
            "trainer_sensitivity": float(event.trainer_sensitivity),
            "natural_outcome": str(event.natural_outcome),
            "technical_outcome": str(event.technical_outcome),
            "target_radius_counts": float(event.target_radius_counts),
            "target_rel_presentation_x": float(presentation_target[0]),
            "target_rel_presentation_y": float(presentation_target[1]),
            "target_rel_generation_recorded_x": float(event.relative_target_x_counts),
            "target_rel_generation_recorded_y": float(event.relative_target_y_counts),
            "initial_distance_recorded": float(event.initial_distance_counts),
            "initial_distance_presentation_derived": float(
                np.linalg.norm(presentation_target)
            ),
            "presentation_minus_generation_x": float(presentation_shift[0]),
            "presentation_minus_generation_y": float(presentation_shift[1]),
            "presentation_minus_generation_norm": float(
                np.linalg.norm(presentation_shift)
            ),
            "presentation_minus_generation_radius_fraction": float(
                np.linalg.norm(presentation_shift)
                / max(float(event.target_radius_counts), 1e-9)
            ),
            "first_presented_unix_ns": first_presented_ns,
            "event_start_unix_ns": event_start_ns,
            "natural_resolution_unix_ns": resolution_ns_optional,
            "tail_end_unix_ns": tail_end_ns,
            "clock_fit_warning_mask": int(event.clock_fit_warning_mask),
            "inside_total_ms_recorded": float(event.inside_total_ms),
            "maximum_consecutive_inside_ms_recorded": float(event.maximum_consecutive_inside_ms),
            "observed_tail_ms_recorded": float(event.observed_tail_ms),
            "click_count": int(event.click_count),
        }
        preflight_reasons: list[str] = []
        if first_presented_ns is None:
            preflight_reasons.append("missing_first_presented")
        if event_start_ns is None:
            preflight_reasons.append("missing_event_start")
        if resolution_ns_optional is None:
            preflight_reasons.append("missing_resolution")
        if (
            event_start_ns is not None
            and first_presented_ns is not None
            and event_start_ns != first_presented_ns
        ):
            preflight_reasons.append("presentation_start_mismatch")
        if not (
            np.isfinite(presentation_target).all()
            and np.isfinite(generation_target).all()
            and np.isfinite(float(event.target_radius_counts))
            and float(event.target_radius_counts) > 0.0
        ):
            preflight_reasons.append("missing_presentation_geometry")
        if (
            event_start_ns is not None
            and resolution_ns_optional is not None
            and resolution_ns_optional <= event_start_ns
        ):
            preflight_reasons.append("invalid_event_interval")
        if (
            event_start_ns is not None
            and resolution_ns_optional is not None
            and (
                len(begin) == 0
                or event_start_ns < int(begin[0])
                or resolution_ns_optional > int(end[-1])
            )
        ):
            preflight_reasons.append("event_outside_dense_coverage")
        preflight_reasons = sorted(set(preflight_reasons))
        if preflight_reasons:
            terminal_reasons = list(preflight_reasons)
            if str(event.natural_outcome) not in {"hit_click", "hit_dwell"}:
                terminal_reasons.append("not_success")
            if str(event.technical_outcome) not in {"", "none"}:
                terminal_reasons.append("technical_outcome")
            if int(event.clock_fit_warning_mask) != 0:
                terminal_reasons.append("clock_fit_warning")
            terminal_reasons = sorted(set(terminal_reasons))
            base.update(
                {
                    "event_usable": False,
                    "rejection_reasons": "|".join(terminal_reasons),
                    "event_end_source": "none",
                }
            )
            reason_counts.update(terminal_reasons)
            rows.append(base)
            continue

        assert event_start_ns is not None and resolution_ns_optional is not None
        start_ns = event_start_ns
        resolution_ns = resolution_ns_optional
        dense = dense_slice_indices(begin, end, start_ns, resolution_ns, policy=slice_policy)
        start_index, stop_index = dense.start, dense.stop
        end_source = f"{slice_policy}_raw_input_resolution"
        usb_click_offset_ms: float | None = None
        usb_click_capture_ns: int | None = None
        usb_click_bin_index: int | None = None
        usb_click_matched: bool | None = None
        click_like = str(event.natural_outcome) in {"hit_click", "miss_click"}
        if prefer_usb_click and click_like:
            edge_position = int(matched_edge_positions[event_row_index])
            usb_click_matched = edge_position >= 0
            if edge_position >= 0:
                click_index = int(usb_edge_rows[edge_position])
                usb_click_capture_ns = int(usb_edge_times[edge_position])
                usb_click_offset_ms = (
                    float(usb_click_capture_ns - resolution_ns) / 1_000_000.0
                )
                if click_index >= start_index:
                    usb_click_bin_index = click_index
                    stop_index = click_index + 1
                    end_source = "authoritative_usb_button_down"
                else:
                    usb_click_matched = False

        event_raw = raw_all[start_index:stop_index]
        target_at_presentation = presentation_target
        tail_stop_index = stop_index
        if tail_end_ns is not None and len(end):
            tail_stop_index = int(
                np.searchsorted(end, tail_end_ns, side="right")
            )
            tail_stop_index = min(max(tail_stop_index, stop_index), len(raw_all))
        post_c_raw = raw_all[stop_index : min(tail_stop_index, stop_index + 64)]
        target_rel_at_c = target_at_presentation - (
            event_raw.sum(axis=0) if len(event_raw) else np.zeros(2, dtype=np.float64)
        )
        tail_metrics = post_c_tail_metrics(
            post_c_raw,
            target_rel_at_c,
            target_at_presentation,
            float(event.target_radius_counts),
        )
        onset = detect_movement_onset(
            event_raw,
            target_at_presentation,
            config=onset_policy,
        )
        if onset is not None:
            movement_before_a = event_raw[: onset.index].sum(axis=0)
            target = target_at_presentation - movement_before_a
            model_start_index = start_index + onset.index
            model_raw = event_raw[onset.index :]
        else:
            target = target_at_presentation
            model_start_index = start_index
            model_raw = event_raw
        quality = shot_quality(model_raw, target, float(event.target_radius_counts))
        reasons = list(
            shot_filter_reasons(
                quality,
                outcome=str(event.natural_outcome),
                technical_outcome=str(event.technical_outcome),
                policy=policy,
            )
        )
        event_quality_mask = int(np.bitwise_or.reduce(quality_mask[start_index:stop_index], initial=0))
        if event_quality_mask != 0:
            reasons.append("capture_quality_mask")
        if int(event.clock_fit_warning_mask) != 0:
            reasons.append("clock_fit_warning")
        if (
            abs(
                float(np.linalg.norm(presentation_target))
                - float(event.initial_distance_counts)
            )
            > 1e-6
        ):
            reasons.append("presentation_geometry_contract_error")
        if (
            float(
                np.linalg.norm(
                    generation_target
                    - np.asarray(
                        [
                            event.relative_target_x_counts,
                            event.relative_target_y_counts,
                        ],
                        dtype=np.float64,
                    )
                )
            )
            > 1e-6
        ):
            reasons.append("generation_geometry_contract_error")
        if stop_index <= start_index:
            reasons.append("empty_event")
        if onset is None:
            reasons.append("no_aligned_movement_onset")
        if prefer_usb_click and click_like and not usb_click_matched:
            reasons.append("unmatched_usb_resolution_click")
        if str(event.natural_outcome) == "hit_click" and int(event.click_count) != 1:
            reasons.append("early_or_multiple_clicks")
        if str(event.natural_outcome) == "hit_dwell" and int(event.click_count) != 0:
            reasons.append("unexpected_click_during_dwell")
        reasons = sorted(set(reasons))
        reason_counts.update(reasons)

        raw_input_endpoint = _raw_input_click_endpoint(event)
        usb_endpoint = event_raw.sum(axis=0) if len(event_raw) else np.zeros(2, dtype=np.float64)
        raw_input_endpoint_error = (
            float(np.linalg.norm(usb_endpoint - raw_input_endpoint))
            if raw_input_endpoint is not None
            else float("nan")
        )
        initial_distance_derived = float(np.linalg.norm(target_at_presentation))
        base.update(asdict(quality))
        base.update(tail_metrics)
        base.update(
            {
                "event_usable": len(reasons) == 0,
                "rejection_reasons": "|".join(reasons),
                "event_end_source": end_source,
                "dense_start_index": start_index,
                "dense_model_start_index": model_start_index,
                "dense_stop_index": stop_index,
                "movement_onset_index": onset.index if onset else -1,
                "movement_onset_threshold": onset.threshold if onset else float("nan"),
                "movement_onset_reason": onset.reason if onset else "not_found",
                "target_rel_at_A_x": float(target[0]),
                "target_rel_at_A_y": float(target[1]),
                "usb_click_offset_ms": usb_click_offset_ms,
                "usb_click_capture_unix_ns": usb_click_capture_ns,
                "usb_click_bin_index": usb_click_bin_index,
                "usb_click_matched": usb_click_matched,
                "usb_rawinput_endpoint_error_counts": raw_input_endpoint_error,
                "initial_distance_derived": initial_distance_derived,
                "generation_target_contract_error": float(
                    np.linalg.norm(
                        generation_target
                        - np.asarray(
                            [
                                event.relative_target_x_counts,
                                event.relative_target_y_counts,
                            ],
                            dtype=np.float64,
                        )
                    )
                ),
                "presentation_distance_contract_error": abs(
                    float(np.linalg.norm(presentation_target))
                    - float(event.initial_distance_counts)
                ),
                "decoded_report_count": int(report_count[model_start_index:stop_index].sum()),
                "zero_delta_report_count": int(zero_reports[model_start_index:stop_index].sum()),
                "dense_nonzero_tick_count": int(np.any(model_raw != 0.0, axis=1).sum()),
                "event_quality_mask": event_quality_mask,
            }
        )
        for threshold, b_config in zip(progress_thresholds, b_configs):
            key = f"b_edge_{int(round(100 * threshold)):02d}"
            decision = edge_progress_decision(
                model_raw,
                target,
                float(event.target_radius_counts),
                b_config=b_config,
                min_prefix_ms=eligibility.min_prefix_ms,
                min_future_ms=eligibility.min_future_ms,
            )
            seam = decision.seam
            base[f"{key}_fired"] = seam is not None
            base[f"{key}_available"] = decision.training_eligible
            base[f"{key}_decision"] = decision.reason
            base[f"{key}_eligibility_reasons"] = "|".join(
                decision.eligibility_reasons
            )
            base[f"{key}_split_index"] = seam.split_index if seam else -1
            base[f"{key}_future_ms"] = len(model_raw) - seam.split_index if seam else -1
            base[f"{key}_realized_progress"] = seam.realized_progress if seam else float("nan")
            base[f"{key}_center_progress"] = seam.center_progress if seam else float("nan")
            base[f"{key}_late_jump"] = bool(
                seam is not None and seam.realized_progress > 0.92
            )
        rows.append(base)

    frame = pd.DataFrame(rows)
    # A single missing value otherwise makes pandas infer float64 and silently
    # rounds ~1.8e18 Unix-ns integers.  Preserve exact boundary timestamps in
    # every returned/profiled table.
    for column in (
        "first_presented_unix_ns",
        "event_start_unix_ns",
        "natural_resolution_unix_ns",
        "tail_end_unix_ns",
        "usb_click_capture_unix_ns",
    ):
        if column in frame:
            # Rebuild from the original Python ints.  Converting the already
            # inferred float64 column would merely preserve its rounded value.
            frame[column] = pd.array(
                [row.get(column, pd.NA) for row in rows],
                dtype="Int64",
            )
    summary = {
        "schema": PROFILE_SCHEMA,
        "source_export": str(Path(export_dir).resolve()),
        "source_export_manifest_sha256": export_manifest_sha256,
        "source_mouse_1ms_sha256": mouse_sha256,
        "source_trainer_events_sha256": trainer_events_sha256,
        "source_session": manifest["source_session"],
        "adapter_id": manifest["adapter_id"],
        "slice_policy": slice_policy,
        "prefer_usb_click": bool(prefer_usb_click),
        "usb_click_button_mask": int(usb_click_button_mask),
        "filter_policy": asdict(policy),
        "progress_thresholds": list(progress_thresholds),
        "causal_seam_contract": contract_record,
        "events_total": int(len(frame)),
        "events_usable": int(frame["event_usable"].fillna(False).sum()),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
    }
    for threshold in progress_thresholds:
        key = f"b_edge_{int(round(100 * threshold)):02d}_available"
        summary[f"{key}_count"] = int(frame.get(key, pd.Series(dtype=bool)).fillna(False).sum())
    return frame, summary


__all__ = [
    "EXPORT_SCHEMA",
    "ADAPTER_ID",
    "PROFILE_SCHEMA",
    "load_export_manifest",
    "load_export_tables",
    "nearest_usb_button_down",
    "exact_usb_button_down_edges",
    "monotone_button_matches",
    "profile_export_session",
]
