"""Leakage-aware evaluation protocols for ABCurves.

The package deliberately keeps four different questions separate:

``floors``
    How far apart are two samples of genuine human movement?
``labeled``
    Can an offline classifier separate labeled human and generated samples?
``cold``
    Can a detector attribute an unknown person's bag without seeing that
    person's clean movement first?
``warm``
    What can be diagnosed when a trusted clean reference from the same source
    is explicitly available?

They are related measurements, not interchangeable claims.
"""

from .bundle import DescriptorBundle, load_descriptor_bundle, write_descriptor_bundle
from .cold import cold_leave_key_out_report, cold_smoke_report
from .floors import human_distance_floor_report, standardized_panel_w1
from .labeled import labeled_c2st_report
from .warm import warm_reference_held_report, warm_smoke_report

__all__ = [
    "DescriptorBundle",
    "load_descriptor_bundle",
    "write_descriptor_bundle",
    "standardized_panel_w1",
    "human_distance_floor_report",
    "labeled_c2st_report",
    "cold_leave_key_out_report",
    "cold_smoke_report",
    "warm_reference_held_report",
    "warm_smoke_report",
]
