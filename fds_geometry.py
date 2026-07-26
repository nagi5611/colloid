# fds_geometry.py
"""Obstruction cross-sections for overlay on a PBX (constant-X) slice view."""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUTF8", "1")

from dataclasses import dataclass
from pathlib import Path

from fdsreader import Simulation

from fds_slice import _resolve_smv_path, load_simulation


@dataclass(frozen=True)
class YzObstRect:
    y_min_m: float
    y_max_m: float
    z_min_m: float
    z_max_m: float
    outline_only: bool


def load_obstruction_yz_from_sim(
    sim: Simulation,
    x_plane_m: float,
    plane_thickness_m: float = 0.25,
) -> list[YzObstRect]:
    """Return Y-Z rectangles where OBST blocks meet the slice plane."""
    x_lo = x_plane_m - plane_thickness_m / 2.0
    x_hi = x_plane_m + plane_thickness_m / 2.0

    rects: list[YzObstRect] = []
    for obst in sim.obstructions:
        if obst.color_index == -2:
            continue
        outline_only = obst.block_type == 2
        for sub in obst._all_subobstructions:
            extent = sub.extent
            if extent.x_end < x_lo or extent.x_start > x_hi:
                continue
            if extent.y_end <= extent.y_start or extent.z_end <= extent.z_start:
                continue
            rects.append(
                YzObstRect(
                    y_min_m=float(extent.y_start),
                    y_max_m=float(extent.y_end),
                    z_min_m=float(extent.z_start),
                    z_max_m=float(extent.z_end),
                    outline_only=outline_only,
                )
            )
    return rects


def load_obstruction_yz_at_x(
    data_dir: Path,
    x_plane_m: float,
    plane_thickness_m: float = 0.25,
) -> list[YzObstRect]:
    """Load simulation and return obstruction cross-sections at x."""
    sim = load_simulation(data_dir)
    return load_obstruction_yz_from_sim(sim, x_plane_m, plane_thickness_m)
