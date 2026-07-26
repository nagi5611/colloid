# temperature_slice_viewer.py
"""Pygame viewer for 30 C temperature bands on an FDS PBX slice."""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pygame

from detect_uniform_temp import (
    TempSegment,
    find_uniform_temp_segments,
    segment_profile_mean,
)
from fds_geometry import YzObstRect, load_obstruction_yz_from_sim
from fds_slice import (
    ScalarSliceField,
    SliceField,
    load_pbx_scalar_from_sim,
    load_pbx_temperature_from_sim,
    load_simulation,
)

TEMP_COLOR_MIN = 20.0
TEMP_COLOR_MAX = 200.0

WINDOW_WIDTH = 1720
WINDOW_HEIGHT = 720
MARGIN = 48
HUD_HEIGHT = 160
PROFILE_HEIGHT = 200
RULER_LEFT_WIDTH = 56
RULER_BOTTOM_HEIGHT = 36
SEGMENT_PANEL_WIDTH = 500
PANEL_GAP = 12
PAINT_BRUSH_RADIUS_PX = 14

TARGET_TEMP_MIN_C = 20.0
TARGET_TEMP_STEP_C = 5.0
TARGET_TEMP_MAX_C = 200.0


def snap_target_temperature_c(value: float) -> float:
    stepped = TARGET_TEMP_MIN_C + round(
        (value - TARGET_TEMP_MIN_C) / TARGET_TEMP_STEP_C
    ) * TARGET_TEMP_STEP_C
    return max(TARGET_TEMP_MIN_C, min(TARGET_TEMP_MAX_C, stepped))


def step_target_temperature_c(value: float, delta_steps: int) -> float:
    return snap_target_temperature_c(value + delta_steps * TARGET_TEMP_STEP_C)


def format_sig2(value: float) -> str:
    """Format a number with two significant figures."""
    if not np.isfinite(value):
        return "n/a"
    return format(float(value), ".2g")


EditMode = Literal["off", "brush", "rect"]


def field_cell_screen_rect(
    iy: int,
    iz: int,
    map_rect: pygame.Rect,
    field: SliceField,
) -> pygame.Rect:
    """Screen rectangle covering one temperature grid cell."""
    ny, nz = field.temperature.shape
    _y_lo, _y_hi, _span, y_increasing = _y_axis_bounds(field)
    screen_ix = iy if y_increasing else ny - 1 - iy
    screen_iz = nz - 1 - iz
    fx0 = screen_ix / ny
    fx1 = (screen_ix + 1) / ny
    fy0 = screen_iz / nz
    fy1 = (screen_iz + 1) / nz
    left = map_rect.left + int(fx0 * map_rect.width)
    right = map_rect.left + int(fx1 * map_rect.width)
    top = map_rect.top + int(fy0 * map_rect.height)
    bottom = map_rect.top + int(fy1 * map_rect.height)
    return pygame.Rect(left, top, max(1, right - left), max(1, bottom - top))


def apply_exclusion_screen_rect(
    mask: np.ndarray,
    field: SliceField,
    map_rect: pygame.Rect,
    p0: tuple[int, int],
    p1: tuple[int, int],
    excluded: bool,
) -> None:
    """Set exclusion for all grid cells overlapping a screen drag rectangle."""
    sel = pygame.Rect(p0[0], p0[1], 0, 0)
    sel.union_ip(pygame.Rect(p1[0], p1[1], 0, 0))
    sel = sel.clip(map_rect)
    if sel.width <= 0 or sel.height <= 0:
        return
    ny, nz = mask.shape
    for iy in range(ny):
        for iz in range(nz):
            cell = field_cell_screen_rect(iy, iz, map_rect, field)
            if sel.colliderect(cell):
                mask[iy, iz] = excluded


def parse_args() -> argparse.Namespace:
    default_data = Path(__file__).resolve().parent / "data"
    parser = argparse.ArgumentParser(description="FDS PBX slice 30 C band viewer")
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--time", type=float, default=600.0)
    parser.add_argument("--target", type=float, default=30.0)
    parser.add_argument(
        "--mean-tolerance",
        type=float,
        default=0.5,
        help="Max |mean(T) - target| for a candidate window (C)",
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=10.0,
        help="Sliding window length along Y (m)",
    )
    parser.add_argument(
        "--y-m",
        type=float,
        default=None,
        help="Keep only segments whose Y range contains this coordinate (m)",
    )
    parser.add_argument(
        "--no-geometry",
        action="store_true",
        help="Do not overlay FDS obstruction geometry on the slice map",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Load data and print summary without opening a window",
    )
    return parser.parse_args()


def temp_to_rgb(value: float) -> tuple[int, int, int]:
    if not np.isfinite(value):
        return (40, 40, 40)
    t = (float(value) - TEMP_COLOR_MIN) / (TEMP_COLOR_MAX - TEMP_COLOR_MIN)
    t = max(0.0, min(1.0, t))
    r = int(255 * t)
    g = int(80 * (1.0 - abs(t - 0.5) * 2))
    b = int(255 * (1.0 - t))
    return (r, g, b)


def _y_axis_bounds(field: SliceField) -> tuple[float, float, float, bool]:
    """Return y_lo (left edge), y_hi (right edge), span, and index-order flag."""
    y0 = float(field.y_coords[0])
    y1 = float(field.y_coords[-1])
    if y0 <= y1:
        return y0, y1, y1 - y0, True
    return y1, y0, y0 - y1, False


def _y_to_fraction(y_m: float, field: SliceField) -> float:
    """Map physical Y to [0, 1] with origin (0 on axis) at the left edge."""
    _y_lo, _y_hi, span, increasing = _y_axis_bounds(field)
    if span <= 0:
        return 0.0
    if increasing:
        return (float(y_m) - _y_lo) / span
    return (_y_hi - float(y_m)) / span


def build_heatmap_surface(field: SliceField) -> pygame.Surface:
    ny, nz = field.temperature.shape
    surface = pygame.Surface((ny, nz))
    _y_lo, _y_hi, _span, y_increasing = _y_axis_bounds(field)
    for iy in range(ny):
        screen_ix = iy if y_increasing else ny - 1 - iy
        for iz in range(nz):
            # Screen Y grows downward; map Z upward (matches world_to_screen).
            screen_iz = nz - 1 - iz
            surface.set_at(
                (screen_ix, screen_iz),
                temp_to_rgb(field.temperature[iy, iz]),
            )
    return surface


def screen_to_field_indices(
    sx: int,
    sy: int,
    rect: pygame.Rect,
    field: SliceField,
) -> tuple[int, int] | None:
    """Map screen pixel on the heatmap to temperature grid (iy, iz)."""
    if not rect.collidepoint(sx, sy):
        return None
    ny, nz = field.temperature.shape
    frac_x = (sx - rect.left) / max(rect.width, 1)
    frac_y = (sy - rect.top) / max(rect.height, 1)
    surf_x = int(frac_x * ny)
    surf_x = max(0, min(ny - 1, surf_x))
    surf_y = int(frac_y * nz)
    surf_y = max(0, min(nz - 1, surf_y))
    _y_lo, _y_hi, _span, y_increasing = _y_axis_bounds(field)
    iy = surf_x if y_increasing else ny - 1 - surf_x
    iz = nz - 1 - surf_y
    return iy, iz


def apply_paint_brush(
    mask: np.ndarray,
    field: SliceField,
    map_rect: pygame.Rect,
    sx: int,
    sy: int,
    radius_px: int,
    excluded: bool,
) -> None:
    """Paint or erase exclusion on the grid under the brush."""
    center = screen_to_field_indices(sx, sy, map_rect, field)
    if center is None:
        return
    iy0, iz0 = center
    ny, nz = mask.shape
    cell_w = map_rect.width / max(ny, 1)
    cell_h = map_rect.height / max(nz, 1)
    riy = max(1, int(np.ceil(radius_px / max(cell_w, 1e-6))))
    riz = max(1, int(np.ceil(radius_px / max(cell_h, 1e-6))))
    for iy in range(max(0, iy0 - riy), min(ny, iy0 + riy + 1)):
        for iz in range(max(0, iz0 - riz), min(nz, iz0 + riz + 1)):
            dy = (iy - iy0) / riy
            dz = (iz - iz0) / riz
            if dy * dy + dz * dz <= 1.0:
                mask[iy, iz] = excluded


def draw_exclusion_overlay(
    screen: pygame.Surface,
    map_rect: pygame.Rect,
    field: SliceField,
    exclusion_mask: np.ndarray,
) -> None:
    """Tint cells excluded from segment search."""
    if not np.any(exclusion_mask):
        return
    ny, nz = exclusion_mask.shape
    for iy in range(ny):
        for iz in range(nz):
            if not exclusion_mask[iy, iz]:
                continue
            cell = field_cell_screen_rect(iy, iz, map_rect, field)
            tile = pygame.Surface((cell.width, cell.height), pygame.SRCALPHA)
            tile.fill((200, 60, 80, 140))
            screen.blit(tile, cell.topleft)


def world_to_screen(
    y_m: float,
    z_m: float,
    rect: pygame.Rect,
    field: SliceField,
) -> tuple[int, int]:
    _y_lo, _y_hi, y_span, _inc = _y_axis_bounds(field)
    z_min, z_max = float(field.z_coords[0]), float(field.z_coords[-1])
    if z_max == z_min:
        z_max = z_min + 1.0
    sx = rect.left + int(_y_to_fraction(y_m, field) * rect.width)
    sy = rect.bottom - int((z_m - z_min) / (z_max - z_min) * rect.height)
    return sx, sy


def _nice_tick_step(span: float, max_ticks: int = 8) -> float:
    if span <= 0:
        return 1.0
    raw = span / max_ticks
    magnitude = 10.0 ** np.floor(np.log10(raw))
    for mult in (1.0, 2.0, 5.0, 10.0):
        step = mult * magnitude
        if span / step <= max_ticks:
            return step
    return magnitude * 10.0


def _format_length_m(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def draw_map_rulers(
    screen: pygame.Surface,
    map_rect: pygame.Rect,
    field: SliceField,
    font: pygame.font.Font,
) -> None:
    """Draw horizontal (Y width) and vertical (Z height) scales in meters."""
    y_lo, y_hi, y_span, _y_inc = _y_axis_bounds(field)
    z_min = float(field.z_coords[0])
    z_max = float(field.z_coords[-1])
    z_span = z_max - z_min if z_max != z_min else 1.0

    tick_color = (140, 140, 150)
    text_color = (190, 190, 200)

    bottom_y = map_rect.bottom + 2
    ruler_h = RULER_BOTTOM_HEIGHT
    pygame.draw.line(
        screen,
        tick_color,
        (map_rect.left, bottom_y),
        (map_rect.right, bottom_y),
        1,
    )
    y_step = _nice_tick_step(y_span)
    y_tick = 0.0
    while y_tick <= y_span + y_step * 0.01:
        frac = y_tick / y_span
        x = map_rect.left + int(frac * map_rect.width)
        pygame.draw.line(screen, tick_color, (x, bottom_y), (x, bottom_y + 6), 1)
        label = font.render(f"{_format_length_m(float(y_tick))} m", True, text_color)
        screen.blit(label, (x - label.get_width() // 2, bottom_y + 8))
        y_tick += y_step

    width_label = font.render(f"Width: {y_span:.1f} m (Y, origin at left)", True, text_color)
    screen.blit(
        width_label,
        (map_rect.centerx - width_label.get_width() // 2, bottom_y + ruler_h - 14),
    )

    left_x = map_rect.left - 2
    pygame.draw.line(
        screen,
        tick_color,
        (left_x, map_rect.top),
        (left_x, map_rect.bottom),
        1,
    )
    z_step = _nice_tick_step(z_span)
    z_tick = np.ceil(z_min / z_step) * z_step
    while z_tick <= z_max + z_step * 0.01:
        frac = (float(z_tick) - z_min) / z_span
        y = map_rect.bottom - int(frac * map_rect.height)
        pygame.draw.line(screen, tick_color, (left_x - 6, y), (left_x, y), 1)
        label = font.render(f"{_format_length_m(float(z_tick))} m", True, text_color)
        label_x = map_rect.left - RULER_LEFT_WIDTH + 4
        screen.blit(label, (label_x, y - label.get_height() // 2))
        z_tick += z_step

    height_label = font.render(f"Height (Z): {z_span:.1f} m", True, text_color)
    screen.blit(
        height_label,
        (map_rect.left - RULER_LEFT_WIDTH + 2, map_rect.top - 20),
    )


def draw_obstruction_geometry(
    screen: pygame.Surface,
    map_rect: pygame.Rect,
    field: SliceField,
    rects: list[YzObstRect],
    enabled: bool,
) -> None:
    if not enabled:
        return
    for rect in rects:
        top_left = world_to_screen(rect.y_min_m, rect.z_max_m, map_rect, field)
        bottom_right = world_to_screen(rect.y_max_m, rect.z_min_m, map_rect, field)
        left = min(top_left[0], bottom_right[0])
        right = max(top_left[0], bottom_right[0])
        top = min(top_left[1], bottom_right[1])
        bottom = max(top_left[1], bottom_right[1])
        width = max(1, right - left)
        height = max(1, bottom - top)
        if rect.outline_only:
            pygame.draw.rect(screen, (210, 175, 90), (left, top, width, height), 2)
            continue
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((180, 140, 70, 100))
        screen.blit(overlay, (left, top))
        pygame.draw.rect(screen, (210, 175, 90), (left, top, width, height), 1)


def _z_half_band_m(field: SliceField, z_idx: int) -> float:
    z_coords = field.z_coords
    if len(z_coords) <= 1:
        return 0.05
    if z_idx <= 0:
        return float(z_coords[1] - z_coords[0]) / 2.0
    if z_idx >= len(z_coords) - 1:
        return float(z_coords[-1] - z_coords[-2]) / 2.0
    return float(z_coords[z_idx + 1] - z_coords[z_idx - 1]) / 2.0


def _segment_map_rect(
    seg: TempSegment,
    field: SliceField,
    map_rect: pygame.Rect,
) -> pygame.Rect:
    dz = _z_half_band_m(field, seg.z_idx)
    y_a = min(seg.y_start_m, seg.y_end_m)
    y_b = max(seg.y_start_m, seg.y_end_m)
    top_left = world_to_screen(y_a, seg.z_m + dz, map_rect, field)
    bottom_right = world_to_screen(y_b, seg.z_m - dz, map_rect, field)
    left = min(top_left[0], bottom_right[0])
    right = max(top_left[0], bottom_right[0])
    top = min(top_left[1], bottom_right[1])
    bottom = max(top_left[1], bottom_right[1])
    return pygame.Rect(left, top, max(1, right - left), max(1, bottom - top)).clip(map_rect)


def draw_segments(
    screen: pygame.Surface,
    field: SliceField,
    segments: list[TempSegment],
    map_rect: pygame.Rect,
    selected: int,
    font: pygame.font.Font,
) -> None:
    """Highlight detected segments on the temperature distribution map."""
    for idx, seg in enumerate(segments):
        band = _segment_map_rect(seg, field, map_rect)
        if band.width <= 0 or band.height <= 0:
            continue
        is_selected = idx == selected
        fill = (255, 255, 0, 110) if is_selected else (0, 255, 255, 75)
        overlay = pygame.Surface((band.width, band.height), pygame.SRCALPHA)
        overlay.fill(fill)
        screen.blit(overlay, band.topleft)
        border_color = (255, 255, 255) if is_selected else (0, 220, 220)
        width = 3 if is_selected else 2
        pygame.draw.rect(screen, border_color, band, width)

        x0, y_mid = world_to_screen(seg.y_start_m, seg.z_m, map_rect, field)
        x1, _ = world_to_screen(seg.y_end_m, seg.z_m, map_rect, field)
        pygame.draw.circle(screen, border_color, (x0, y_mid), 4)
        pygame.draw.circle(screen, border_color, (x1, y_mid), 4)

        label = font.render(
            f"#{idx + 1}  L={seg.length_m:.1f}m  mean {seg.mean_temp_c:.2f}C  "
            f"dT {seg.spread_c:.2f}C",
            True,
            (255, 255, 255) if is_selected else (200, 255, 255),
        )
        lx = band.centerx - label.get_width() // 2
        ly = band.top - label.get_height() - 2
        if ly < map_rect.top:
            ly = band.bottom + 2
        lx = max(map_rect.left, min(lx, map_rect.right - label.get_width()))
        screen.blit(label, (lx, ly))


def draw_profile(
    screen: pygame.Surface,
    field: SliceField,
    segment: TempSegment | None,
    profile_rect: pygame.Rect,
    font: pygame.font.Font,
    target_c: float,
    tolerance_c: float,
) -> None:
    pygame.draw.rect(screen, (30, 30, 30), profile_rect)
    pygame.draw.rect(screen, (80, 80, 80), profile_rect, 1)
    label = font.render("Temperature vs Y (selected segment)", True, (220, 220, 220))
    screen.blit(label, (profile_rect.left + 8, profile_rect.top + 4))

    if segment is not None:
        err_text = font.render(
            f"Segment error (max-min): {segment.spread_c:.3f} C",
            True,
            (180, 220, 180),
        )
        screen.blit(err_text, (profile_rect.right - err_text.get_width() - 8, profile_rect.top + 4))

    if segment is None:
        msg = font.render("No segment selected", True, (180, 180, 180))
        screen.blit(msg, (profile_rect.centerx - 80, profile_rect.centery))
        return

    ys = field.y_coords[segment.y_start_idx : segment.y_end_idx + 1]
    vals = field.temperature[segment.y_start_idx : segment.y_end_idx + 1, segment.z_idx]
    if len(ys) < 2:
        return

    plot = profile_rect.inflate(-16, -32)
    y_min, y_max = float(ys[0]), float(ys[-1])
    t_min = min(float(np.nanmin(vals)), target_c - tolerance_c - 2)
    t_max = max(float(np.nanmax(vals)), target_c + tolerance_c + 2)
    if y_max == y_min:
        y_max = y_min + 1.0
    if t_max == t_min:
        t_max = t_min + 1.0

    band_top = plot.bottom - int((target_c + tolerance_c - t_min) / (t_max - t_min) * plot.height)
    band_bottom = plot.bottom - int((target_c - tolerance_c - t_min) / (t_max - t_min) * plot.height)
    pygame.draw.rect(
        screen,
        (60, 90, 60),
        pygame.Rect(plot.left, band_top, plot.width, max(1, band_bottom - band_top)),
    )

    ref_y = plot.bottom - int((target_c - t_min) / (t_max - t_min) * plot.height)
    pygame.draw.line(screen, (200, 200, 100), (plot.left, ref_y), (plot.right, ref_y), 1)

    points: list[tuple[int, int]] = []
    for y_val, t_val in zip(ys, vals):
        if not np.isfinite(t_val):
            continue
        px = plot.left + int((float(y_val) - y_min) / (y_max - y_min) * plot.width)
        py = plot.bottom - int((float(t_val) - t_min) / (t_max - t_min) * plot.height)
        points.append((px, py))
    if len(points) >= 2:
        pygame.draw.lines(screen, (100, 200, 255), False, points, 2)


def print_summary(
    field: SliceField,
    segments: list[TempSegment],
    target_c: float,
    tolerance_c: float,
) -> None:
    finite = field.temperature[np.isfinite(field.temperature)]
    tmin = float(np.min(finite)) if finite.size else float("nan")
    tmax = float(np.max(finite)) if finite.size else float("nan")
    print(f"Time: {field.time_s:.1f} s | Slice: PBX=0 ({field.slice_id}) at x={field.x_plane_m:.3f} m")
    print(f"Field temperature range: {tmin:.2f} - {tmax:.2f} C")
    print(
        f"10 m sliding windows with |mean(T)-{target_c}| <= {tolerance_c} C: {len(segments)}"
    )
    if segments:
        best = segments[0]
        print(
            f"Best segment: length={best.length_m:.2f} m, "
            f"Y=[{best.y_start_m:.2f}, {best.y_end_m:.2f}] m, "
            f"Z={best.z_m:.2f} m, mean T={best.mean_temp_c:.3f} C, "
            f"|mean-{target_c}|={best.score:.4f} C, spread (max-min)={best.spread_c:.4f} C"
        )
    else:
        print("No segments in band (see HUD in viewer).")


def segment_table_row_height() -> int:
    return 26


def segment_table_header_height() -> int:
    return 92


def table_visible_row_count(table_rect: pygame.Rect) -> int:
    body_h = table_rect.height - segment_table_header_height()
    return max(1, body_h // segment_table_row_height())


def clamp_table_scroll_row(
    scroll_row: int,
    segment_count: int,
    table_rect: pygame.Rect,
) -> int:
    visible = table_visible_row_count(table_rect)
    max_scroll = max(0, segment_count - visible)
    return max(0, min(scroll_row, max_scroll))


def segment_table_row_at(
    pos: tuple[int, int],
    table_rect: pygame.Rect,
    row_count: int,
    scroll_row: int,
) -> int | None:
    if row_count <= 0 or not table_rect.collidepoint(pos):
        return None
    body_top = table_rect.top + segment_table_header_height()
    if pos[1] < body_top:
        return None
    local_row = (pos[1] - body_top) // segment_table_row_height()
    row = scroll_row + local_row
    if row < 0 or row >= row_count:
        return None
    visible = table_visible_row_count(table_rect)
    if local_row >= visible:
        return None
    return int(row)


def draw_segment_table(
    screen: pygame.Surface,
    table_rect: pygame.Rect,
    segments: list[TempSegment],
    target_c: float,
    soot_od_field: ScalarSliceField | None,
    soot_mass_fraction_field: ScalarSliceField | None,
    selected: int,
    scroll_row: int,
    font_title: pygame.font.Font,
    font: pygame.font.Font,
) -> None:
    """Right-hand panel: all segments ranked by temperature error."""
    pygame.draw.rect(screen, (24, 24, 28), table_rect)
    pygame.draw.rect(screen, (90, 90, 100), table_rect, 1)

    title = font_title.render(
        "All segments (lowest |mean - target| first)",
        True,
        (220, 220, 230),
    )
    screen.blit(title, (table_rect.left + 10, table_rect.top + 8))

    od_unit = soot_od_field.quantity_unit if soot_od_field else "1/m"
    mf_unit = (
        soot_mass_fraction_field.quantity_unit
        if soot_mass_fraction_field
        else "kg/kg"
    )
    if soot_od_field is not None:
        line1 = (
            f"OD: {soot_od_field.slice_id} ({od_unit}); "
            f"smoke: mass fraction Y_s ({mf_unit}), not per m³"
        )
    else:
        line1 = "Soot optical density: not loaded"
    screen.blit(font.render(line1, True, (150, 155, 165)), (table_rect.left + 10, table_rect.top + 28))
    if soot_mass_fraction_field is not None:
        line2 = f"Y_s from {soot_mass_fraction_field.slice_id} (soot mass / mixture mass)"
        screen.blit(font.render(line2, True, (140, 145, 155)), (table_rect.left + 10, table_rect.top + 46))
    else:
        screen.blit(
            font.render("Soot mass fraction: not loaded", True, (150, 120, 120)),
            (table_rect.left + 10, table_rect.top + 46),
        )

    od_grid = soot_od_field.values if soot_od_field is not None else None
    mf_grid = soot_mass_fraction_field.values if soot_mass_fraction_field is not None else None
    headers = (
        "#",
        "mean T",
        "|err|",
        f"OD ({od_unit})",
        f"Y_s ({mf_unit})",
    )
    col_x = (
        table_rect.left + 6,
        table_rect.left + 34,
        table_rect.left + 108,
        table_rect.left + 178,
        table_rect.left + 288,
    )
    header_y = table_rect.top + 66
    for label, x in zip(headers, col_x):
        screen.blit(font.render(label, True, (170, 175, 185)), (x, header_y))
    pygame.draw.line(
        screen,
        (70, 70, 80),
        (table_rect.left + 6, header_y + 18),
        (table_rect.right - 6, header_y + 18),
        1,
    )

    row_h = segment_table_row_height()
    body_top = table_rect.top + segment_table_header_height()
    body_height = max(0, table_rect.bottom - body_top)
    body_rect = pygame.Rect(table_rect.left, body_top, table_rect.width, body_height)
    visible_rows = table_visible_row_count(table_rect)
    first_row = clamp_table_scroll_row(scroll_row, len(segments), table_rect)

    prev_clip = screen.get_clip()
    screen.set_clip(body_rect)
    for idx in range(first_row, min(len(segments), first_row + visible_rows + 1)):
        seg = segments[idx]
        row_y = body_top + (idx - first_row) * row_h
        if row_y + row_h > body_rect.bottom:
            break
        row_rect = pygame.Rect(table_rect.left + 4, row_y, table_rect.width - 8, row_h - 2)
        if idx == selected:
            pygame.draw.rect(screen, (55, 70, 90), row_rect, border_radius=3)
        mean_od = segment_profile_mean(od_grid, seg) if od_grid is not None else float("nan")
        mean_mf = segment_profile_mean(mf_grid, seg) if mf_grid is not None else float("nan")
        cells = (
            str(idx + 1),
            format_sig2(seg.mean_temp_c),
            format_sig2(seg.score),
            format_sig2(mean_od),
            format_sig2(mean_mf),
        )
        text_color = (240, 245, 255) if idx == selected else (200, 205, 215)
        for text, x in zip(cells, col_x):
            screen.blit(font.render(text, True, text_color), (x, row_y + 4))
    screen.set_clip(prev_clip)

    segment_count = len(segments)
    max_scroll = clamp_table_scroll_row(999999, segment_count, table_rect)
    if max_scroll > 0:
        track = pygame.Rect(table_rect.right - 10, body_top + 2, 6, body_height - 4)
        pygame.draw.rect(screen, (45, 45, 52), track, border_radius=3)
        thumb_h = max(20, int(body_height * visible_rows / max(segment_count, 1)))
        thumb_y = body_top + int(
            (body_height - thumb_h) * (first_row / max_scroll) if max_scroll else 0
        )
        pygame.draw.rect(screen, (110, 115, 130), (track.left, thumb_y, track.width, thumb_h))

    if not segments:
        msg = font.render("No segments in band", True, (140, 140, 150))
        screen.blit(msg, (table_rect.left + 12, body_top + 8))


def compute_view_layout(
    win_w: int,
    win_h: int,
) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
    """Map, profile, and segment table regions for the current window size."""
    map_top = MARGIN
    map_left = MARGIN + RULER_LEFT_WIDTH
    map_width = max(
        200,
        win_w - map_left - MARGIN - SEGMENT_PANEL_WIDTH - PANEL_GAP,
    )
    map_height = max(
        120,
        win_h - HUD_HEIGHT - PROFILE_HEIGHT - 2 * MARGIN - RULER_BOTTOM_HEIGHT,
    )
    map_rect = pygame.Rect(map_left, map_top, map_width, map_height)
    profile_rect = pygame.Rect(
        MARGIN,
        map_rect.bottom + RULER_BOTTOM_HEIGHT + 8,
        max(100, map_rect.right - MARGIN),
        PROFILE_HEIGHT,
    )
    table_rect = pygame.Rect(
        map_rect.right + PANEL_GAP,
        map_top,
        SEGMENT_PANEL_WIDTH,
        max(120, profile_rect.bottom - map_top),
    )
    return map_rect, profile_rect, table_rect


def _try_maximize_window() -> None:
    try:
        from pygame._sdl2 import video

        video.Window.from_display_module().maximize()
    except (ImportError, AttributeError, TypeError, pygame.error):
        pass


def create_maximized_screen() -> pygame.Surface:
    """Open a resizable window and maximize it (fallback: desktop resolution)."""
    screen = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        pygame.RESIZABLE,
    )
    _try_maximize_window()
    pygame.display.flip()
    size = screen.get_size()
    if size[0] > 0 and size[1] > 0:
        return screen
    info = pygame.display.Info()
    fallback_w = max(WINDOW_WIDTH, int(info.current_w))
    fallback_h = max(WINDOW_HEIGHT, int(info.current_h))
    return pygame.display.set_mode((fallback_w, fallback_h), pygame.RESIZABLE)


def toggle_fullscreen_display(
    screen: pygame.Surface,
    fullscreen: bool,
    windowed_size: tuple[int, int],
) -> tuple[pygame.Surface, bool, tuple[int, int]]:
    """Toggle F11-style fullscreen; return updated screen, flag, and saved window size."""
    if fullscreen:
        screen = pygame.display.set_mode(windowed_size, pygame.RESIZABLE)
        _try_maximize_window()
        return screen, False, windowed_size
    windowed_size = screen.get_size()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    return screen, True, windowed_size


def run_pygame(
    field: SliceField,
    initial_target_c: float,
    mean_tolerance_c: float,
    segment_length_m: float,
    geometry_rects: list[YzObstRect],
    show_geometry: bool,
    soot_od_field: ScalarSliceField | None,
    soot_mass_fraction_field: ScalarSliceField | None,
    y_filter_m: float | None = None,
) -> None:
    pygame.init()
    screen = create_maximized_screen()
    fullscreen = False
    windowed_size = screen.get_size()
    font = pygame.font.SysFont("Segoe UI", 18)
    small = pygame.font.SysFont("Segoe UI", 16)

    map_rect, profile_rect, table_rect = compute_view_layout(*screen.get_size())

    target_c = snap_target_temperature_c(initial_target_c)
    ny, nz = field.temperature.shape
    exclusion_mask = np.zeros((ny, nz), dtype=bool)
    edit_mode: EditMode = "off"
    paint_drag = False
    paint_erase = False
    segments_dirty = False
    rect_drag = False
    rect_start: tuple[int, int] | None = None
    rect_current: tuple[int, int] | None = None
    table_scroll_row = 0

    def recompute_segments() -> list[TempSegment]:
        return find_uniform_temp_segments(
            field,
            target_c=target_c,
            mean_tolerance_c=mean_tolerance_c,
            segment_length_m=segment_length_m,
            y_filter_m=y_filter_m,
            exclusion_mask=exclusion_mask,
        )

    def ensure_table_row_visible() -> None:
        nonlocal table_scroll_row
        visible = table_visible_row_count(table_rect)
        if segments and selected < table_scroll_row:
            table_scroll_row = selected
        elif segments and selected >= table_scroll_row + visible:
            table_scroll_row = selected - visible + 1
        table_scroll_row = clamp_table_scroll_row(
            table_scroll_row, len(segments), table_rect
        )

    segments = recompute_segments()
    selected = 0
    geometry_on = show_geometry

    heat = build_heatmap_surface(field)
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                if not fullscreen:
                    screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                    windowed_size = event.size
                    map_rect, profile_rect, table_rect = compute_view_layout(*event.size)
            elif event.type == pygame.WINDOWSIZECHANGED:
                size = getattr(event, "size", None) or screen.get_size()
                map_rect, profile_rect, table_rect = compute_view_layout(*size)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_F11:
                    screen, fullscreen, windowed_size = toggle_fullscreen_display(
                        screen, fullscreen, windowed_size
                    )
                    map_rect, profile_rect, table_rect = compute_view_layout(
                        *screen.get_size()
                    )
                elif event.key == pygame.K_p:
                    if edit_mode == "off":
                        edit_mode = "brush"
                    elif edit_mode == "brush":
                        edit_mode = "rect"
                    else:
                        edit_mode = "off"
                        rect_drag = False
                        rect_start = None
                        rect_current = None
                elif event.key == pygame.K_c:
                    if np.any(exclusion_mask):
                        exclusion_mask[:] = False
                        segments = recompute_segments()
                        selected = 0
                        table_scroll_row = 0
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_RIGHTBRACKET):
                    new_target = step_target_temperature_c(target_c, 1)
                    if new_target != target_c:
                        target_c = new_target
                        segments = recompute_segments()
                        selected = 0
                        table_scroll_row = 0
                elif event.key in (pygame.K_MINUS, pygame.K_LEFTBRACKET):
                    new_target = step_target_temperature_c(target_c, -1)
                    if new_target != target_c:
                        target_c = new_target
                        segments = recompute_segments()
                        selected = 0
                        table_scroll_row = 0
                elif event.key == pygame.K_SPACE and segments:
                    selected = (selected + 1) % len(segments)
                    ensure_table_row_visible()
                elif pygame.K_1 <= event.key <= pygame.K_9 and segments:
                    idx = event.key - pygame.K_1
                    if idx < len(segments):
                        selected = idx
                        ensure_table_row_visible()
                elif event.key == pygame.K_0 and len(segments) >= 10:
                    selected = 9
                    ensure_table_row_visible()
                elif event.key == pygame.K_g:
                    geometry_on = not geometry_on
            elif event.type == pygame.MOUSEWHEEL:
                if table_rect.collidepoint(pygame.mouse.get_pos()):
                    table_scroll_row = clamp_table_scroll_row(
                        table_scroll_row - event.y,
                        len(segments),
                        table_rect,
                    )
            elif event.type == pygame.MOUSEBUTTONDOWN:
                row = segment_table_row_at(
                    event.pos, table_rect, len(segments), table_scroll_row
                )
                if row is not None and event.button == 1:
                    selected = row
                elif edit_mode == "brush" and map_rect.collidepoint(event.pos):
                    paint_drag = True
                    paint_erase = event.button == 3 or (
                        event.button == 1
                        and pygame.key.get_mods() & pygame.KMOD_SHIFT
                    )
                    apply_paint_brush(
                        exclusion_mask,
                        field,
                        map_rect,
                        event.pos[0],
                        event.pos[1],
                        PAINT_BRUSH_RADIUS_PX,
                        excluded=not paint_erase,
                    )
                    segments_dirty = True
                elif (
                    edit_mode == "rect"
                    and event.button in (1, 3)
                    and map_rect.collidepoint(event.pos)
                ):
                    rect_drag = True
                    rect_start = event.pos
                    rect_current = event.pos
                    paint_erase = event.button == 3 or (
                        event.button == 1
                        and pygame.key.get_mods() & pygame.KMOD_SHIFT
                    )
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (1, 3):
                    if edit_mode == "rect" and rect_drag and rect_start is not None:
                        end_pos = event.pos
                        apply_exclusion_screen_rect(
                            exclusion_mask,
                            field,
                            map_rect,
                            rect_start,
                            end_pos,
                            excluded=not paint_erase,
                        )
                        segments = recompute_segments()
                        selected = min(selected, max(0, len(segments) - 1))
                        table_scroll_row = clamp_table_scroll_row(
                            table_scroll_row, len(segments), table_rect
                        )
                        ensure_table_row_visible()
                        rect_drag = False
                        rect_start = None
                        rect_current = None
                    elif edit_mode == "brush" and paint_drag and segments_dirty:
                        segments = recompute_segments()
                        selected = min(selected, max(0, len(segments) - 1))
                        table_scroll_row = clamp_table_scroll_row(
                            table_scroll_row, len(segments), table_rect
                        )
                        ensure_table_row_visible()
                    paint_drag = False
                    segments_dirty = False
            elif event.type == pygame.MOUSEMOTION:
                if edit_mode == "brush" and paint_drag and map_rect.collidepoint(event.pos):
                    apply_paint_brush(
                        exclusion_mask,
                        field,
                        map_rect,
                        event.pos[0],
                        event.pos[1],
                        PAINT_BRUSH_RADIUS_PX,
                        excluded=not paint_erase,
                    )
                    segments_dirty = True
                elif edit_mode == "rect" and rect_drag:
                    rect_current = event.pos

        caption_suffix = ""
        if edit_mode == "brush":
            caption_suffix = " | BRUSH exclude"
        elif edit_mode == "rect":
            caption_suffix = " | RECT exclude"
        pygame.display.set_caption(
            f"FDS Temperature Slice — mean target {target_c:.0f} C{caption_suffix}"
        )

        screen.fill((18, 18, 22))
        scaled = pygame.transform.smoothscale(heat, (map_rect.width, map_rect.height))
        screen.blit(scaled, map_rect)
        pygame.draw.rect(screen, (120, 120, 120), map_rect, 1)

        draw_exclusion_overlay(screen, map_rect, field, exclusion_mask)
        if edit_mode == "brush" and map_rect.collidepoint(pygame.mouse.get_pos()):
            mx, my = pygame.mouse.get_pos()
            pygame.draw.circle(
                screen,
                (255, 120, 140) if not paint_erase else (120, 200, 255),
                (mx, my),
                PAINT_BRUSH_RADIUS_PX,
                2,
            )
        if (
            edit_mode == "rect"
            and rect_drag
            and rect_start is not None
            and rect_current is not None
        ):
            preview = pygame.Rect(rect_start[0], rect_start[1], 0, 0)
            preview.union_ip(pygame.Rect(rect_current[0], rect_current[1], 0, 0))
            preview = preview.clip(map_rect)
            edge = (255, 120, 140) if not paint_erase else (120, 200, 255)
            pygame.draw.rect(screen, edge, preview, 2)
            overlay = pygame.Surface((preview.width, preview.height), pygame.SRCALPHA)
            fill = (200, 60, 80, 60) if not paint_erase else (80, 140, 200, 50)
            overlay.fill(fill)
            screen.blit(overlay, preview.topleft)

        draw_obstruction_geometry(screen, map_rect, field, geometry_rects, geometry_on)
        draw_segments(screen, field, segments, map_rect, selected, small)
        draw_map_rulers(screen, map_rect, field, small)
        table_scroll_row = clamp_table_scroll_row(
            table_scroll_row, len(segments), table_rect
        )
        draw_segment_table(
            screen,
            table_rect,
            segments,
            target_c,
            soot_od_field,
            soot_mass_fraction_field,
            selected,
            table_scroll_row,
            font,
            small,
        )
        seg = segments[selected] if segments else None
        draw_profile(screen, field, seg, profile_rect, font, target_c, mean_tolerance_c)

        hud_y = profile_rect.bottom + 8
        lines = [
            f"Time: {field.time_s:.1f} s",
            f"Slice: PBX=0 ({field.slice_id})",
            f"Window: {segment_length_m:.1f} m | mean T {target_c:.1f} C (+/-{mean_tolerance_c:.1f} C)",
            (
                f"Segments: {len(segments)} (table: wheel to scroll)"
                if segments
                else "No segments in band"
            ),
        ]
        if seg is not None:
            lines.append(
                f"Selected: L={seg.length_m:.1f} m, |mean-{target_c}|={seg.score:.3f} C, "
                f"spread={seg.spread_c:.3f} C"
            )
        geom_state = "on" if geometry_on else "off"
        lines.append(f"Geometry overlay: {geom_state} ({len(geometry_rects)} OBST) | G: toggle")
        excluded_cells = int(np.sum(exclusion_mask))
        edit_labels = {"off": "off", "brush": "brush", "rect": "rectangle"}
        lines.append(
            f"Search exclude: {edit_labels[edit_mode]} | excluded cells: {excluded_cells} | "
            f"P: off -> brush -> rect | drag: exclude | Shift+drag or RMB: erase | C: clear"
        )
        lines.append(
            f"Target mean: +/- or [/]  (20 C, step {TARGET_TEMP_STEP_C:.0f} C, "
            f"max {TARGET_TEMP_MAX_C:.0f} C)"
        )
        lines.append("Space: next | 1-9,0: pick | click table row | F11: fullscreen | Q: Quit")
        for i, text in enumerate(lines):
            screen.blit(small.render(text, True, (210, 210, 210)), (MARGIN, hud_y + i * 18))

        axis = small.render("Map: Y (m, 0 at left), Z vertical (m)", True, (160, 160, 160))
        screen.blit(axis, (map_rect.left, map_rect.top - 20))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        if data_dir.name == "dat":
            suggested = data_dir.parent / "data"
            if suggested.is_dir():
                print(
                    f"Data directory not found: {data_dir} (did you mean '{suggested}'?)",
                    file=sys.stderr,
                )
                return 1
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    try:
        sim = load_simulation(data_dir)
        field = load_pbx_temperature_from_sim(sim, time_s=args.time)
        try:
            soot_od_field = load_pbx_scalar_from_sim(
                sim,
                slice_id="Temp slice.011",
                quantity_name="SOOT OPTICAL DENSITY",
                time_s=args.time,
            )
            print(
                f"Soot optical density: {soot_od_field.slice_id} at t={soot_od_field.time_s:.1f} s "
                f"({soot_od_field.quantity_unit})"
            )
        except ValueError as exc:
            soot_od_field = None
            print(f"Warning: soot optical density not loaded: {exc}", file=sys.stderr)
        try:
            soot_mass_fraction_field = load_pbx_scalar_from_sim(
                sim,
                slice_id="Temp slice.006",
                quantity_name="SOOT MASS FRACTION",
                time_s=args.time,
            )
            print(
                f"Soot mass fraction: {soot_mass_fraction_field.slice_id} at "
                f"t={soot_mass_fraction_field.time_s:.1f} s "
                f"({soot_mass_fraction_field.quantity_unit})"
            )
        except ValueError as exc:
            soot_mass_fraction_field = None
            print(f"Warning: soot mass fraction not loaded: {exc}", file=sys.stderr)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    initial_target = snap_target_temperature_c(args.target)
    segments = find_uniform_temp_segments(
        field,
        target_c=initial_target,
        mean_tolerance_c=args.mean_tolerance,
        segment_length_m=args.segment_length,
        y_filter_m=args.y_m,
    )
    print_summary(field, segments, initial_target, args.mean_tolerance)

    geometry_rects: list[YzObstRect] = []
    if not args.no_geometry:
        geometry_rects = load_obstruction_yz_from_sim(sim, field.x_plane_m)
        print(f"Obstruction geometry at slice plane: {len(geometry_rects)} rectangles")

    if args.headless:
        return 0

    run_pygame(
        field,
        initial_target,
        args.mean_tolerance,
        args.segment_length,
        geometry_rects,
        show_geometry=not args.no_geometry,
        soot_od_field=soot_od_field,
        soot_mass_fraction_field=soot_mass_fraction_field,
        y_filter_m=args.y_m,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
