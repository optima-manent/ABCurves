"""B2-23 CPU inference. Importing this module does not load research or Torch."""

__version__ = "0.1.0"


def load(assets, *, seed=7, initial_xy=(0., 0.), history=None, diagnostics=False,
         backend="native", kernel_backend="numba", prewarm=True, skip_unused=True):
    """Load a verified exported bundle and create one independent stream.

    Initialization/prewarm is synchronous. ``advance`` never sleeps or performs
    device I/O; its output coordinates are common counts at 1 ms endpoints.
    """
    from .planner import Planner
    from .runtime import MovementRuntime
    planner = Planner(assets, diagnostics=diagnostics, backend=backend,
                      kernel_backend=kernel_backend, skip_unused=skip_unused)
    if prewarm:
        planner.prewarm()
    return MovementRuntime(planner, initial_xy=initial_xy, history=history, seed=seed)
