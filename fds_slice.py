# fds_slice.py
"""Load FDS PBX temperature slice data via fdsreader."""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUTF8", "1")

import builtins
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_real_open = builtins.open


def _open_with_utf8_fallback(
    file,
    mode: str = "r",
    buffering: int = -1,
    encoding: str | None = None,
    errors: str | None = None,
    **kwargs,
):
    if "b" not in mode and encoding is None:
        encoding = "utf-8"
        errors = errors or "replace"
    return _real_open(file, mode, buffering, encoding=encoding, errors=errors, **kwargs)


builtins.open = _open_with_utf8_fallback

from fdsreader import Simulation


@dataclass(frozen=True)
class SliceField:
    """2D temperature on a PBX slice: axes are physical Y and Z."""

    temperature: np.ndarray
    y_coords: np.ndarray
    z_coords: np.ndarray
    time_s: float
    slice_id: str
    x_plane_m: float


@dataclass(frozen=True)
class ScalarSliceField:
    """2D scalar quantity on a PBX slice (same Y-Z layout as temperature)."""

    values: np.ndarray
    y_coords: np.ndarray
    z_coords: np.ndarray
    time_s: float
    slice_id: str
    quantity_name: str
    quantity_unit: str
    x_plane_m: float


def _resolve_smv_path(data_dir: Path) -> Path:
    data_dir = data_dir.resolve()
    smv = data_dir / "colloid04.smv"
    if smv.is_file():
        return smv
    matches = list(data_dir.glob("*.smv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No .smv file found under '{data_dir}'")
    raise FileNotFoundError(f"Multiple .smv files under '{data_dir}'; specify one explicitly")


def _find_temp_pbx_slice(sim: Simulation):
    return _find_pbx_slice(sim, slice_id="Temp slice", quantity_name="TEMPERATURE")


def _find_pbx_slice(
    sim: Simulation,
    slice_id: str,
    quantity_name: str | None = None,
):
    matches = []
    for slc in sim.slices:
        if slc.id != slice_id:
            continue
        if slc.orientation != 1:
            continue
        if quantity_name is not None and slc.quantity.name != quantity_name:
            continue
        matches.append(slc)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        hint = f" id='{slice_id}'"
        if quantity_name:
            hint += f" quantity='{quantity_name}'"
        raise ValueError(f"PBX slice not found:{hint}")
    raise ValueError(
        f"Multiple PBX slices match id='{slice_id}'"
        + (f" quantity='{quantity_name}'" if quantity_name else "")
    )


def _load_pbx_field_from_slice(slc, time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    time_index = slc.get_nearest_timestep(time_s)
    actual_time = float(slc.times[time_index])
    grid, coords = slc.to_global(return_coordinates=True)
    field = np.asarray(grid[time_index], dtype=np.float64)
    if slc.orientation != 1:
        raise ValueError(f"Expected PBX slice (orientation=1), got {slc.orientation}")
    if field.ndim != 2:
        raise ValueError(f"Expected 2D PBX field, got shape {field.shape}")
    y_coords = np.asarray(coords["y"], dtype=np.float64)
    z_coords = np.asarray(coords["z"], dtype=np.float64)
    x_plane = float(coords["x"][0]) if len(coords["x"]) else 0.0
    if field.shape != (len(y_coords), len(z_coords)):
        raise ValueError(
            f"Field shape {field.shape} does not match coordinates "
            f"({len(y_coords)}, {len(z_coords)})"
        )
    return field, y_coords, z_coords, actual_time, x_plane


def load_simulation(data_dir: Path) -> Simulation:
    """Load the FDS simulation once (slice + geometry should share this)."""
    return Simulation(str(_resolve_smv_path(data_dir)))


def load_pbx_temperature_from_sim(sim: Simulation, time_s: float = 600.0) -> SliceField:
    """Load stitched PBX temperature field at the nearest output time."""
    slc = _find_temp_pbx_slice(sim)
    field, y_coords, z_coords, actual_time, x_plane = _load_pbx_field_from_slice(slc, time_s)
    return SliceField(
        temperature=field,
        y_coords=y_coords,
        z_coords=z_coords,
        time_s=actual_time,
        slice_id=slc.id,
        x_plane_m=x_plane,
    )


def load_pbx_scalar_from_sim(
    sim: Simulation,
    slice_id: str,
    quantity_name: str,
    time_s: float = 600.0,
) -> ScalarSliceField:
    """Load a PBX scalar slice (e.g. soot optical density on Temp slice.011)."""
    slc = _find_pbx_slice(sim, slice_id=slice_id, quantity_name=quantity_name)
    field, y_coords, z_coords, actual_time, x_plane = _load_pbx_field_from_slice(slc, time_s)
    unit = getattr(slc.quantity, "unit", "") or ""
    return ScalarSliceField(
        values=field,
        y_coords=y_coords,
        z_coords=z_coords,
        time_s=actual_time,
        slice_id=slc.id,
        quantity_name=slc.quantity.name,
        quantity_unit=str(unit),
        x_plane_m=x_plane,
    )


def slice_field_from_scalar(scalar: ScalarSliceField) -> SliceField:
    """Reuse slice viewer helpers (heatmap, geometry) with a scalar PBX field."""
    return SliceField(
        temperature=np.asarray(scalar.values, dtype=np.float64),
        y_coords=np.asarray(scalar.y_coords, dtype=np.float64),
        z_coords=np.asarray(scalar.z_coords, dtype=np.float64),
        time_s=scalar.time_s,
        slice_id=scalar.slice_id,
        x_plane_m=scalar.x_plane_m,
    )


def load_pbx_temperature_slice(data_dir: Path, time_s: float = 600.0) -> SliceField:
    """Convenience wrapper that loads the simulation and reads the PBX slice."""
    return load_pbx_temperature_from_sim(load_simulation(data_dir), time_s=time_s)
