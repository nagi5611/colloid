# detect_uniform_temp.py
"""Find fixed-length Y windows (sliding along Y) whose mean temperature is near a target."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fds_slice import SliceField

MAX_SEGMENT_HITS = 100


@dataclass(frozen=True)
class TempSegment:
    y_start_m: float
    y_end_m: float
    z_m: float
    score: float
    mean_temp_c: float
    spread_c: float
    y_start_idx: int
    y_end_idx: int
    z_idx: int
    length_m: float


def _end_index_for_length(y_coords: np.ndarray, start: int, length_m: float) -> int | None:
    y0 = float(y_coords[start])
    for end in range(start + 1, len(y_coords)):
        if abs(float(y_coords[end]) - y0) >= length_m:
            return end
    return None


def find_uniform_temp_segments(
    field: SliceField,
    target_c: float = 30.0,
    mean_tolerance_c: float = 0.5,
    segment_length_m: float = 10.0,
    y_filter_m: float | None = None,
    exclusion_mask: np.ndarray | None = None,
    max_hits: int = MAX_SEGMENT_HITS,
) -> list[TempSegment]:
    """Slide a fixed-length window along Y; keep means within target ± mean_tolerance."""
    temp = field.temperature
    y_coords = np.asarray(field.y_coords, dtype=np.float64)
    z_coords = field.z_coords
    ny = len(y_coords)

    if exclusion_mask is not None:
        if exclusion_mask.shape != temp.shape:
            raise ValueError(
                f"exclusion_mask shape {exclusion_mask.shape} != temperature {temp.shape}"
            )

    segments: list[TempSegment] = []

    for z_idx, z_m in enumerate(z_coords):
        profile = temp[:, z_idx]
        for start in range(ny):
            end = _end_index_for_length(y_coords, start, segment_length_m)
            if end is None:
                break
            if exclusion_mask is not None and np.any(
                exclusion_mask[start : end + 1, z_idx]
            ):
                continue
            values = profile[start : end + 1]
            if not np.any(np.isfinite(values)):
                continue
            mean_t = float(np.nanmean(values))
            score = abs(mean_t - target_c)
            if score > mean_tolerance_c:
                continue

            y_start_m = float(min(y_coords[start], y_coords[end]))
            y_end_m = float(max(y_coords[start], y_coords[end]))
            if y_filter_m is not None and not (y_start_m <= y_filter_m <= y_end_m):
                continue

            spread = float(np.nanmax(values) - np.nanmin(values))
            length_m = y_end_m - y_start_m
            segments.append(
                TempSegment(
                    y_start_m=y_start_m,
                    y_end_m=y_end_m,
                    z_m=float(z_m),
                    score=score,
                    mean_temp_c=mean_t,
                    spread_c=spread,
                    y_start_idx=start,
                    y_end_idx=end,
                    z_idx=z_idx,
                    length_m=length_m,
                )
            )

    segments.sort(key=lambda s: (s.score, -s.length_m))
    if max_hits > 0:
        segments = segments[:max_hits]
    return segments


def segment_profile_mean(grid: np.ndarray, seg: TempSegment) -> float:
    """Mean of a 2D field along the segment's Y window at fixed Z index."""
    patch = grid[seg.y_start_idx : seg.y_end_idx + 1, seg.z_idx]
    if not np.any(np.isfinite(patch)):
        return float("nan")
    return float(np.nanmean(patch))
