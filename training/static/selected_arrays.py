"""Selected Static Planner sufficient-target construction from canonical raw events.

This module depends only on NumPy and the public ABCurves package. It preserves
the selected training operation order; cuts and source roles come from the
frozen recipe, not a new split or a second eligibility decision.
"""
from __future__ import annotations

import math
import numpy as np

from abcurves.features import summary_features
from abcurves.prodmp import ProDMP, ProDMPConfig
from abcurves.smoothing import smooth_dxdy


class SelectedTargets:
    def __init__(self, config: dict, feature_names: list[str]):
        self.prodmp = ProDMP(ProDMPConfig(
            n_basis=config['n_basis'], alpha=config['alpha'],
            alpha_phase=config['alpha_phase'], ridge=config['weight_ridge']))
        self.ridge = config['weight_ridge']
        self.feature_names = feature_names
        self.maps = {}

    def _fit(self, path, ydot):
        duration = len(path)
        if duration not in self.maps:
            design = np.asarray(self.prodmp.position_basis(duration), dtype=np.float64)
            col_norm = np.linalg.norm(design, axis=0)
            col_norm[col_norm < 1e-9] = 1.0
            scaled = design / col_norm[None, :]
            gram = scaled.T @ scaled + self.ridge * np.eye(scaled.shape[1])
            projection = np.linalg.solve(gram, scaled.T) / col_norm[:, None]
            boundary = self.prodmp.boundary_response(
                np.arange(1, duration + 1, dtype=np.float64), float(duration),
                np.zeros(2), np.asarray([1.0, 0.0]))[:, 0]
            self.maps[duration] = projection, boundary
        projection, boundary = self.maps[duration]
        return projection @ (path - boundary[:, None] * ydot[None, :])

    def row(self, event_raw, cut, *, event_smooth=None):
        raw = np.asarray(event_raw, dtype=np.float32)
        split, duration = int(cut['split_index']), int(cut['future_ms'])
        if not (0 < split < len(raw) and duration == len(raw)-split and 12 <= duration <= 1000):
            raise ValueError('cut duration outside selected contract')
        smooth = (smooth_dxdy(raw, 'triangular_moving_average_path:window=5')
                  if event_smooth is None else np.asarray(event_smooth, dtype=np.float32))
        prefix = raw[:split]
        take = min(160, len(prefix))
        padded = np.zeros((160, 2), dtype=np.float32)
        mask = np.zeros(160, dtype=np.uint8)
        padded[-take:] = prefix[-take:]
        mask[-take:] = 1
        target = np.asarray([cut['target_b_x'], cut['target_b_y']], dtype=np.float64)
        path = np.cumsum(smooth[split:].astype(np.float64), axis=0, dtype=np.float64)
        ydot = prefix[-1].astype(np.float64)
        coefficients = self._fit(path, ydot)
        y_raw = np.r_[coefficients.reshape(-1), math.log(float(duration))].astype(np.float32)
        query = np.arange(1, 49, dtype=np.float64) / 48.0 * float(duration)
        time_axis = np.arange(duration + 1, dtype=np.float64)
        grid = np.column_stack([
            np.interp(query, time_axis, np.r_[0.0, path[:, axis]])
            for axis in range(2)]).astype(np.float32)
        features = summary_features(prefix[-take:], tuple(target), float(cut['target_radius']),
            float(cut['center_progress']), horizon=1000, b_index_ms=float(split))
        return {'prefix_raw_dxdy': padded, 'prefix_mask': mask,
            'summary_features': np.asarray([features[n] for n in self.feature_names], dtype=np.float64),
            'y_raw': y_raw, 'grid_path': grid, 'tau': np.float32(duration),
            'ydot': ydot.astype(np.float32)}
