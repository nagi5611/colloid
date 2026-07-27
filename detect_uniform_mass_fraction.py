# detect_uniform_mass_fraction.py
"""Find fixed-length Y windows whose mean soot mass fraction is near a target."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from detect_uniform_temp import _end_index_for_length, segment_profile_mean
from fds_slice import ScalarSliceField

__all__ = [
    "MfSegment",
    "find_uniform_mass_fraction_segments",
    "segment_profile_mean",
]


@dataclass(frozen=True)
class MfSegment:
    y_start_m: float
    y_end_m: float
    z_m: float
    score: float
    mean_mf: float
    spread_mf: float
    y_start_idx: int
    y_end_idx: int
    z_idx: int
    length_m: float


def find_uniform_mass_fraction_segments(
    field: ScalarSliceField,
    target_mf: float = 0.003,
    mean_tolerance: float = 0.0002,
    segment_length_m: float = 10.0,
    y_filter_m: float | None = None,
    exclusion_mask: np.ndarray | None = None,
) -> list[MfSegment]:
    """Slide a fixed-length window along Y; keep means within target ± mean_tolerance."""
    values = field.values
    y_coords = np.asarray(field.y_coords, dtype=np.float64)
    z_coords = field.z_coords
    ny = len(y_coords)

    if exclusion_mask is not None:
        if exclusion_mask.shape != values.shape:
            raise ValueError(
                f"exclusion_mask shape {exclusion_mask.shape} != field {values.shape}"
            )

    segments: list[MfSegment] = []

    for z_idx, z_m in enumerate(z_coords):
        profile = values[:, z_idx]
        for start in range(ny):
            end = _end_index_for_length(y_coords, start, segment_length_m)
            if end is None:
                break
            if exclusion_mask is not None and np.any(
                exclusion_mask[start : end + 1, z_idx]
            ):
                continue
            window = profile[start : end + 1]
            if not np.any(np.isfinite(window)):
                continue
            mean_mf = float(np.nanmean(window))
            score = abs(mean_mf - target_mf)
            if score > mean_tolerance:
                continue

            y_start_m = float(min(y_coords[start], y_coords[end]))
            y_end_m = float(max(y_coords[start], y_coords[end]))
            if y_filter_m is not None and not (y_start_m <= y_filter_m <= y_end_m):
                continue

            spread_mf = float(np.nanmax(window) - np.nanmin(window))
            length_m = y_end_m - y_start_m
            segments.append(
                MfSegment(
                    y_start_m=y_start_m,
                    y_end_m=y_end_m,
                    z_m=float(z_m),
                    score=score,
                    mean_mf=mean_mf,
                    spread_mf=spread_mf,
                    y_start_idx=start,
                    y_end_idx=end,
                    z_idx=z_idx,
                    length_m=length_m,
                )
            )

    segments.sort(key=lambda s: (s.score, -s.length_m))
    return segments


def max_sliding_window_mean(
    field: ScalarSliceField,
    segment_length_m: float = 10.0,
) -> float:
    """Maximum mean Y_s over any fixed-length Y window (same geometry as segment search)."""
    values = field.values
    y_coords = np.asarray(field.y_coords, dtype=np.float64)
    ny = len(y_coords)
    best = 0.0
    for z_idx in range(values.shape[1]):
        for start in range(ny):
            end = _end_index_for_length(y_coords, start, segment_length_m)
            if end is None:
                break
            mean_mf = float(np.nanmean(values[start : end + 1, z_idx]))
            if np.isfinite(mean_mf) and mean_mf > best:
                best = mean_mf
    return best
