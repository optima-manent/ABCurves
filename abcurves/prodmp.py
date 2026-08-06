"""Probabilistic Dynamic Movement Primitives (ProDMP) core, pure NumPy.

This module implements the trajectory representation from *ProDMP: A Unified
Perspective on Dynamic and Probabilistic Movement Primitives* (Li et al.,
arXiv:2210.01531).  It is intentionally standalone -- it imports nothing from
the rest of ``abcurves`` -- so it can be unit-tested in isolation and reused
as a planner backbone in other projects.

Why ABCurves uses this representation
-------------------------------------
A mouse movement is a 2D time series, and time series are a terrible thing to
predict point-by-point.  Every planner we trained directly on delta streams
either blurred into an average or fell apart on speed/acceleration profiles.
ProDMP fixes the *representation*: solving the DMP ODE analytically yields a
*linear basis function* trajectory

    y(t) = xi1(t) * y_b + xi2(t) * ydot_b + H(t) . w_g

where the boundary state ``(y_b, ydot_b)`` at the cut point B is baked in
(exact velocity continuity at the seam -- no hand-tuned blending), the per-DoF
forcing weights ``w`` express arbitrary shape/curvature, and the goal
coefficient ``g`` provides endpoint convergence.  A whole curve -- any curve a
hand can draw in a few hundred milliseconds -- compresses to ~40 numbers plus
a duration, and those numbers live in a smooth, learnable space.  A Gaussian
over ``w_g`` maps to a Gaussian over trajectories, which is what gives the
planner its "distribution over plausible continuations".

Math summary (critically damped, beta = alpha / 4)
--------------------------------------------------
DMP ODE:  tau^2 y'' = alpha (beta (g - y) - tau y') + f(x),
          f(x) = x * (phi(x) . w) / sum(phi(x)),   x(t) = exp(-alpha_x t / tau).

The homogeneous solutions are ``y1 = e^{-a t}`` and ``y2 = t e^{-a t}`` with
``a = alpha / (2 tau)``.  Solving the inhomogeneous ODE by variation of
parameters and grouping the goal term gives a position basis whose normalized
(phase-time ``s = t / tau``) form is *tau-invariant*:

    Phi_w_i(s) = s e^{-A s} P2_i(s) - e^{-A s} P1_i(s),     A = alpha / 2
    Phi_g(s)   = 1 - e^{-A s} (1 + A s)
    P1_i(s) = int_0^s s' e^{A s'} x(s') phi_tilde_i(s') ds'
    P2_i(s) = int_0^s    e^{A s'} x(s') phi_tilde_i(s') ds'

with analytic phase-derivatives

    dPhi_w_i/ds = e^{-A s} [ (1 - A s) P2_i(s) + A P1_i(s) ]
    dPhi_g/ds   = A^2 s e^{-A s}.

The boundary correction (Eq. idmp_pos_full) uses

    xi1 = (yd2_b y1 - yd1_b y2) / D,   xi2 = (y1_b y2 - y2_b y1) / D,
    xi3 = -xi1,                        xi4 = -xi2,   D = y1_b yd2_b - y2_b yd1_b
    H(t) = Phi(t) + xi3 Phi(t_b) + xi4 Phidot(t_b).

For ABCurves the boundary is the start of the horizon (``t_b = 0``), where
``Phi(0) = 0`` and ``Phidot(0) = 0``, so ``H(t) = Phi(t)`` and the boundary
state enters only through ``xi1 y_b + xi2 ydot_b``.  The general ``t_b`` form
is implemented for faithfulness to the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EPS = 1e-9


@dataclass(frozen=True)
class ProDMPConfig:
    """Configuration for the offline ProDMP basis.

    Attributes:
        n_basis: Number of RBF forcing weights per degree of freedom.  The full
            weight vector ``w_g`` has ``n_basis + 1`` entries (the extra one is
            the goal coefficient ``g``).
        alpha: Spring constant ``alpha`` (``a_z``).  Critically damped, so the
            damping ``beta`` is fixed at ``alpha / 4``.
        alpha_phase: Phase decay ``alpha_x`` in ``x(t) = exp(-alpha_x t / tau)``.
        basis_width_scale: Multiplies the RBF precisions; >1 narrows the bumps.
        grid_points: Resolution of the offline normalized-time integration grid.
        ridge: Default Tikhonov regularization for least-squares weight fits.
    """

    n_basis: int = 20
    alpha: float = 25.0
    alpha_phase: float = 3.0
    basis_width_scale: float = 1.0
    grid_points: int = 2000
    ridge: float = 1e-3

    def __post_init__(self) -> None:
        if self.n_basis < 2:
            raise ValueError("n_basis must be >= 2")
        if self.alpha <= 0 or self.alpha_phase <= 0:
            raise ValueError("alpha and alpha_phase must be positive")
        if self.grid_points < 16:
            raise ValueError("grid_points must be >= 16")


class ProDMP:
    """Offline ProDMP basis with boundary-condition aware generation and fits."""

    def __init__(self, config: ProDMPConfig = ProDMPConfig()) -> None:
        self.config = config
        self.n_basis = int(config.n_basis)
        self.alpha = float(config.alpha)
        self.beta = float(config.alpha) / 4.0
        self.alpha_phase = float(config.alpha_phase)
        self._half_alpha = self.alpha / 2.0
        self._precompute_normalized_basis()
        self._design_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Offline precomputation
    # ------------------------------------------------------------------
    def _precompute_normalized_basis(self) -> None:
        cfg = self.config
        s = np.linspace(0.0, 1.0, int(cfg.grid_points) + 1, dtype=np.float64)
        phase = np.exp(-self.alpha_phase * s)

        centers = np.exp(-self.alpha_phase * (np.arange(self.n_basis, dtype=np.float64) / (self.n_basis - 1)))
        spacing = np.diff(centers)
        widths = np.empty(self.n_basis, dtype=np.float64)
        widths[:-1] = 1.0 / np.maximum(spacing * spacing, EPS)
        widths[-1] = widths[-2]
        widths *= float(cfg.basis_width_scale)

        rbf = np.exp(-widths[None, :] * (phase[:, None] - centers[None, :]) ** 2)
        rbf_sum = np.sum(rbf, axis=1, keepdims=True)
        rbf_sum[rbf_sum < EPS] = EPS
        phi_tilde = rbf / rbf_sum
        forcing = phase[:, None] * phi_tilde  # x(s) * normalized RBF -> [G, N]

        ea = np.exp(self._half_alpha * s)
        em = np.exp(-self._half_alpha * s)
        integ1 = (s[:, None] * ea[:, None]) * forcing
        integ2 = ea[:, None] * forcing
        p1 = _cumulative_trapezoid(integ1, s)
        p2 = _cumulative_trapezoid(integ2, s)

        phi_w = (s * em)[:, None] * p2 - em[:, None] * p1
        phi_g = 1.0 - em * (1.0 + self._half_alpha * s)
        dphi_w = em[:, None] * (((1.0 - self._half_alpha * s)[:, None] * p2) + self._half_alpha * p1)
        dphi_g = (self._half_alpha ** 2) * s * em

        self._s_grid = s
        self._phase_grid = phase
        self._centers = centers
        self._widths = widths
        self._phi_grid = np.concatenate([phi_w, phi_g[:, None]], axis=1)  # [G, N+1]
        self._dphi_grid = np.concatenate([dphi_w, dphi_g[:, None]], axis=1)  # [G, N+1]

    @property
    def n_weights(self) -> int:
        """Length of ``w_g`` per DoF (forcing weights plus goal)."""
        return self.n_basis + 1

    # ------------------------------------------------------------------
    # Basis sampling
    # ------------------------------------------------------------------
    def _basis_at(self, s_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.clip(np.asarray(s_values, dtype=np.float64).reshape(-1), 0.0, 1.0)
        phi = np.empty((len(s), self.n_weights), dtype=np.float64)
        dphi = np.empty((len(s), self.n_weights), dtype=np.float64)
        grid = self._s_grid
        for j in range(self.n_weights):
            phi[:, j] = np.interp(s, grid, self._phi_grid[:, j])
            dphi[:, j] = np.interp(s, grid, self._dphi_grid[:, j])
        return phi, dphi

    def phase(self, duration: int) -> np.ndarray:
        """Phase variable ``x(t)`` over ``t = 1..duration`` with ``tau = duration``."""
        d = int(duration)
        t = np.arange(1, d + 1, dtype=np.float64)
        return np.exp(-self.alpha_phase * t / float(d))

    def position_basis(self, duration: int) -> np.ndarray:
        """Boundary-aware position basis ``H`` of shape ``[duration, n_basis+1]``.

        With ``t_b = 0`` this equals the raw normalized basis sampled at
        ``s = t / duration``; multiplying by ``w_g`` and adding the boundary
        free-response reproduces the trajectory.
        """
        d = int(duration)
        cached = self._design_cache.get(d)
        if cached is None:
            t = np.arange(1, d + 1, dtype=np.float64)
            _, _, cached = self._components(t, float(d), 0.0)
            self._design_cache[d] = cached
        return cached

    def velocity_basis(self, duration: int) -> np.ndarray:
        """Velocity (dy/dt) basis of shape ``[duration, n_basis+1]`` (``t_b = 0``)."""
        d = int(duration)
        t = np.arange(1, d + 1, dtype=np.float64)
        _, dphi = self._basis_at(t / float(d))
        return dphi / float(d)

    # ------------------------------------------------------------------
    # Boundary-condition machinery (Eq. idmp_pos_full)
    # ------------------------------------------------------------------
    def _components(self, t: np.ndarray, tau: float, t_b: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(xi1, xi2, H)`` for actual times ``t`` and boundary ``t_b``."""
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        y1, y2, yd1, yd2 = _critically_damped(t, tau, self.alpha)
        tb = np.asarray([float(t_b)], dtype=np.float64)
        y1b, y2b, yd1b, yd2b = (float(v[0]) for v in _critically_damped(tb, tau, self.alpha))
        det = y1b * yd2b - y2b * yd1b
        if abs(det) < EPS:
            det = EPS if det >= 0 else -EPS
        xi1 = (yd2b * y1 - yd1b * y2) / det
        xi2 = (y1b * y2 - y2b * y1) / det
        xi3 = -xi1
        xi4 = -xi2
        phi_t, _ = self._basis_at(t / tau)
        phi_b, dphi_b = self._basis_at(np.asarray([t_b / tau]))
        phidot_b = dphi_b[0] / tau  # velocity basis at the boundary
        h = phi_t + xi3[:, None] * phi_b[0][None, :] + xi4[:, None] * phidot_b[None, :]
        return xi1, xi2, h

    def boundary_response(
        self,
        t: np.ndarray,
        tau: float,
        y_b: np.ndarray,
        ydot_b: np.ndarray,
        *,
        t_b: float = 0.0,
    ) -> np.ndarray:
        """Free response ``xi1 y_b + xi2 ydot_b`` (no forcing)."""
        xi1, xi2, _ = self._components(t, tau, t_b)
        y_b = _as_dof_vector(y_b)
        ydot_b = _as_dof_vector(ydot_b)
        return xi1[:, None] * y_b[None, :] + xi2[:, None] * ydot_b[None, :]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        t: np.ndarray,
        w_g: np.ndarray,
        *,
        tau: float,
        ydot_b: np.ndarray,
        y_b: np.ndarray | None = None,
        t_b: float = 0.0,
    ) -> np.ndarray:
        """Positions at arbitrary times ``t`` for weights ``w_g`` of shape ``[n_basis+1, dof]``."""
        w_g = np.atleast_2d(np.asarray(w_g, dtype=np.float64))
        if w_g.shape[0] != self.n_weights:
            raise ValueError(f"w_g must have {self.n_weights} rows, got {w_g.shape[0]}")
        dof = w_g.shape[1]
        y_b = np.zeros(dof, dtype=np.float64) if y_b is None else _as_dof_vector(y_b, dof)
        ydot_b = _as_dof_vector(ydot_b, dof)
        xi1, xi2, h = self._components(t, tau, t_b)
        return xi1[:, None] * y_b[None, :] + xi2[:, None] * ydot_b[None, :] + h @ w_g

    def generate(
        self,
        w_g: np.ndarray,
        ydot_b: np.ndarray,
        duration: int,
        *,
        y_b: np.ndarray | None = None,
        goal: np.ndarray | None = None,
        goal_mode: str = "learned",
    ) -> np.ndarray:
        """Generate the B-relative position path of shape ``[duration, dof]``.

        ``goal_mode='target'`` overrides the goal coefficient (last row of
        ``w_g``) with the supplied ``goal`` so the trajectory homes to a known
        target instead of a learned endpoint.
        """
        d = int(duration)
        w_g = np.atleast_2d(np.asarray(w_g, dtype=np.float64)).copy()
        if goal_mode == "target":
            if goal is None:
                raise ValueError("goal_mode='target' requires a goal vector")
            w_g[-1] = _as_dof_vector(goal, w_g.shape[1])
        t = np.arange(1, d + 1, dtype=np.float64)
        return self.evaluate(t, w_g, tau=float(d), ydot_b=ydot_b, y_b=y_b, t_b=0.0)

    def generate_deltas(
        self,
        w_g: np.ndarray,
        ydot_b: np.ndarray,
        duration: int,
        *,
        y_b: np.ndarray | None = None,
        goal: np.ndarray | None = None,
        goal_mode: str = "learned",
    ) -> np.ndarray:
        """Generate the B-relative delta stream of shape ``[duration, dof]``.

        The cumulative sum of the returned deltas reproduces :meth:`generate`.
        """
        positions = self.generate(w_g, ydot_b, duration, y_b=y_b, goal=goal, goal_mode=goal_mode)
        dof = positions.shape[1]
        start = np.zeros(dof, dtype=np.float64) if y_b is None else _as_dof_vector(y_b, dof)
        return np.diff(positions, axis=0, prepend=start[None, :])

    # ------------------------------------------------------------------
    # Fitting (imitation)
    # ------------------------------------------------------------------
    def fit_weights(
        self,
        path_xy: np.ndarray,
        ydot_b: np.ndarray,
        duration: int,
        *,
        y_b: np.ndarray | None = None,
        goal: np.ndarray | None = None,
        goal_mode: str = "learned",
        ridge: float | None = None,
    ) -> np.ndarray:
        """Least-squares fit of ``w_g`` to a B-relative position path.

        ``path_xy`` holds positions at ``t = 1..duration`` (the cumulative sum of
        the future delta stream).  Returns ``w_g`` of shape ``[n_basis+1, dof]``.
        """
        path = np.asarray(path_xy, dtype=np.float64)
        if path.ndim != 2:
            raise ValueError("path_xy must be [duration, dof]")
        d = int(duration)
        path = path[:d]
        dof = path.shape[1]
        lam = self.config.ridge if ridge is None else float(ridge)
        t = np.arange(1, d + 1, dtype=np.float64)
        h = self.position_basis(d)
        residual = path - self.boundary_response(t, float(d), np.zeros(dof) if y_b is None else y_b, ydot_b)
        if goal_mode == "target":
            if goal is None:
                raise ValueError("goal_mode='target' requires a goal vector")
            g = _as_dof_vector(goal, dof)
            residual = residual - h[:, -1:] * g[None, :]
            w = _ridge_solve(h[:, :-1], residual, lam)
            return np.concatenate([w, g[None, :]], axis=0)
        return _ridge_solve(h, residual, lam)

    def reconstruct(
        self,
        path_xy: np.ndarray,
        ydot_b: np.ndarray,
        duration: int,
        *,
        y_b: np.ndarray | None = None,
        goal: np.ndarray | None = None,
        goal_mode: str = "learned",
        ridge: float | None = None,
    ) -> np.ndarray:
        """Fit then regenerate -- the representation ceiling for a given path."""
        w_g = self.fit_weights(path_xy, ydot_b, duration, y_b=y_b, goal=goal, goal_mode=goal_mode, ridge=ridge)
        return self.generate(w_g, ydot_b, duration, y_b=y_b, goal=goal, goal_mode=goal_mode)

    # ------------------------------------------------------------------
    # Probabilistic head (Eq. idmp_mean_cov)
    # ------------------------------------------------------------------
    def trajectory_distribution(
        self,
        mu_w_g: np.ndarray,
        cov_free: np.ndarray,
        ydot_b: np.ndarray,
        duration: int,
        *,
        y_b: np.ndarray | None = None,
        goal_mode: str = "learned",
    ) -> dict[str, np.ndarray]:
        """Map a Gaussian over weights to a Gaussian over the trajectory.

        ``cov_free`` is the covariance over the *free* weights -- all
        ``n_basis+1`` weights for ``goal_mode='learned'`` or the ``n_basis``
        forcing weights for ``goal_mode='target'`` (the goal is then fixed).
        Returns the mean path and the per-time-step standard deviation (shared
        across DoF because a single weight covariance is used).
        """
        d = int(duration)
        mu = np.atleast_2d(np.asarray(mu_w_g, dtype=np.float64))
        dof = mu.shape[1]
        mean = self.generate(mu, ydot_b, d, y_b=y_b, goal_mode="learned")
        h = self.position_basis(d)
        h_free = h if goal_mode == "learned" else h[:, :-1]
        cov_free = np.asarray(cov_free, dtype=np.float64)
        var = np.einsum("ti,ij,tj->t", h_free, cov_free, h_free)
        std = np.sqrt(np.maximum(var, 0.0))
        return {"mean_path": mean, "std_path": np.repeat(std[:, None], dof, axis=1)}

    def sample_delta_streams(
        self,
        mu_w_g: np.ndarray,
        cov_free: np.ndarray,
        ydot_b: np.ndarray,
        duration: int,
        *,
        n_samples: int,
        rng: np.random.Generator,
        y_b: np.ndarray | None = None,
        goal: np.ndarray | None = None,
        goal_mode: str = "learned",
    ) -> np.ndarray:
        """Sample ``n_samples`` B-relative delta streams ``[S, duration, dof]``."""
        d = int(duration)
        mu = np.atleast_2d(np.asarray(mu_w_g, dtype=np.float64)).copy()
        dof = mu.shape[1]
        cov_free = np.asarray(cov_free, dtype=np.float64)
        p_free = cov_free.shape[0]
        chol = _safe_cholesky(cov_free)
        out = np.zeros((int(n_samples), d, dof), dtype=np.float64)
        for s_idx in range(int(n_samples)):
            sample = mu.copy()
            noise = chol @ rng.standard_normal((p_free, dof))
            sample[:p_free] += noise
            out[s_idx] = self.generate_deltas(sample, ydot_b, d, y_b=y_b, goal=goal, goal_mode=goal_mode)
        return out

    # ------------------------------------------------------------------
    # Reference ODE integrator (for validation, not used by the planner)
    # ------------------------------------------------------------------
    def simulate_ode(
        self,
        w: np.ndarray,
        goal: np.ndarray,
        ydot_b: np.ndarray,
        duration: int,
        *,
        y_b: np.ndarray | None = None,
        substeps: int = 20,
    ) -> np.ndarray:
        """Euler-integrate the raw DMP ODE; used to validate the analytic basis.

        ``w`` are the ``n_basis`` forcing weights per DoF; ``goal`` is ``g``.
        """
        d = int(duration)
        tau = float(d)
        w = np.atleast_2d(np.asarray(w, dtype=np.float64))
        dof = w.shape[1]
        g = _as_dof_vector(goal, dof)
        y = np.zeros(dof, dtype=np.float64) if y_b is None else _as_dof_vector(y_b, dof)
        ydot = _as_dof_vector(ydot_b, dof)
        dt = 1.0 / float(substeps)
        positions = np.zeros((d, dof), dtype=np.float64)
        a_x = self.alpha_phase
        for step in range(d * substeps):
            t = step * dt
            x = float(np.exp(-a_x * t / tau))
            rbf = np.exp(-self._widths * (x - self._centers) ** 2)
            rbf_sum = float(np.sum(rbf))
            phi_tilde = rbf / (rbf_sum if rbf_sum > EPS else EPS)
            forcing = x * (phi_tilde @ w)  # [dof]
            ydd = (self.alpha * (self.beta * (g - y) - tau * ydot) + forcing) / (tau * tau)
            ydot = ydot + ydd * dt
            y = y + ydot * dt
            t_next = t + dt
            idx = int(round(t_next)) - 1
            if 0 <= idx < d and abs(t_next - round(t_next)) < dt / 2.0:
                positions[idx] = y
        positions[-1] = y
        return positions


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _critically_damped(t: np.ndarray, tau: float, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = alpha / (2.0 * float(tau))
    e = np.exp(-a * t)
    y1 = e
    y2 = t * e
    yd1 = -a * e
    yd2 = e * (1.0 - a * t)
    return y1, y2, yd1, yd2


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral from ``x[0]`` (output[0] == 0)."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    dx = np.diff(x)
    avg = 0.5 * (y[1:] + y[:-1])
    if y.ndim == 2:
        increments = avg * dx[:, None]
    else:
        increments = avg * dx
    out = np.zeros_like(y)
    out[1:] = np.cumsum(increments, axis=0)
    return out


def _ridge_solve(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    """Column-normalized ridge least squares.

    The ProDMP position basis is severely ill-conditioned -- late basis columns
    are tiny because the phase forcing ``x`` decays toward the goal -- so a raw
    ridge penalizes columns unequally and yields unstable weights.  Normalizing
    each column to unit norm before the solve makes the penalty uniform and
    recovers near-exact reconstructions, then we rescale back to the raw basis.
    """
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    col_norm = np.linalg.norm(design, axis=0)
    col_norm[col_norm < EPS] = 1.0
    scaled = design / col_norm[None, :]
    p = scaled.shape[1]
    gram = scaled.T @ scaled + float(ridge) * np.eye(p, dtype=np.float64)
    solution = np.linalg.solve(gram, scaled.T @ target)
    return solution / col_norm[:, None]


def _as_dof_vector(value: np.ndarray, dof: int | None = None) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64)).reshape(-1)
    if dof is not None and arr.shape[0] != dof:
        if arr.shape[0] == 1:
            arr = np.repeat(arr, dof)
        else:
            raise ValueError(f"expected {dof} DoF, got {arr.shape[0]}")
    return arr


def _safe_cholesky(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=np.float64)
    p = cov.shape[0]
    jitter = 1e-12
    for _ in range(8):
        try:
            return np.linalg.cholesky(cov + jitter * np.eye(p))
        except np.linalg.LinAlgError:
            jitter *= 10.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-10, None)
    return eigvecs @ np.diag(np.sqrt(eigvals))


def weight_residual_covariance(residuals: np.ndarray, *, shrinkage: float = 1e-3) -> np.ndarray:
    """Shrinkage covariance of stacked per-DoF weight residuals ``[m, p]``."""
    res = np.asarray(residuals, dtype=np.float64)
    if res.ndim != 2 or res.shape[0] < 2:
        p = res.shape[1] if res.ndim == 2 else int(res.shape[-1])
        return np.eye(p, dtype=np.float64) * float(shrinkage)
    cov = np.cov(res, rowvar=False)
    cov = np.atleast_2d(cov)
    trace_mean = float(np.trace(cov) / cov.shape[0])
    return (1.0 - shrinkage) * cov + shrinkage * trace_mean * np.eye(cov.shape[0])
