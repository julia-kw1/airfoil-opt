"""geometric utility functions for airfoil analysis."""
import numpy as np
from typing import Tuple


def normalize_unit_chord(coords: np.ndarray) -> np.ndarray:
    """scale an airfoil so its chord length equals one."""
    x = coords[:, 0]
    x_min = float(np.min(x))
    chord = float(np.max(x) - x_min)
    if chord <= 1e-6:
        return coords.copy()

    normalized = coords.copy()
    normalized[:, 0] = (normalized[:, 0] - x_min) / chord
    normalized[:, 1] = normalized[:, 1] / chord
    return normalized


def split_upper_lower(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """split coordinates into upper and lower curves, both trailing->leading."""
    i_le = int(np.argmin(coords[:, 0]))
    upper = coords[:i_le + 1][::-1]
    lower = coords[i_le:]
    return upper, lower


def thickness_ratio(coords: np.ndarray) -> float:
    """maximum thickness-to-chord ratio."""
    upper, lower = split_upper_lower(coords)
    x_min = max(float(np.min(upper[:, 0])), float(np.min(lower[:, 0])))
    x_max = min(float(np.max(upper[:, 0])), float(np.max(lower[:, 0])))

    if x_max <= x_min:
        return np.nan

    x_sample = np.linspace(x_min, x_max, 200)
    y_upper = np.interp(x_sample, upper[:, 0], upper[:, 1])
    y_lower = np.interp(x_sample, lower[:, 0], lower[:, 1])
    return float(np.max(y_upper - y_lower))


def extract_camberline(coords: np.ndarray, n_points: int = 200) -> np.ndarray:
    """camber line sampled across the chord."""
    upper, lower = split_upper_lower(coords)
    x_common = np.linspace(0.0, 1.0, n_points)
    y_upper = np.interp(x_common, upper[:, 0], upper[:, 1], left=np.nan, right=np.nan)
    y_lower = np.interp(x_common, lower[:, 0], lower[:, 1], left=np.nan, right=np.nan)

    mask = np.isfinite(y_upper) & np.isfinite(y_lower)
    return np.column_stack((x_common[mask], 0.5 * (y_upper[mask] + y_lower[mask])))


def has_self_intersection(coords: np.ndarray) -> bool:
    """return true when the polyline intersects itself."""
    def ccw(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c):
        return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
                min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

    def segments_intersect(p1, p2, p3, p4):
        o1, o2 = ccw(p1, p2, p3), ccw(p1, p2, p4)
        o3, o4 = ccw(p3, p4, p1), ccw(p3, p4, p2)

        if o1 == 0 and on_segment(p1, p2, p3):
            return True
        if o2 == 0 and on_segment(p1, p2, p4):
            return True
        if o3 == 0 and on_segment(p3, p4, p1):
            return True
        if o4 == 0 and on_segment(p3, p4, p2):
            return True

        return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

    n_pts = coords.shape[0]
    for i in range(n_pts - 1):
        seg_start = coords[i]
        seg_end = coords[i + 1]
        for j in range(i + 2, n_pts - 1):
            if i == 0 and j + 1 == n_pts - 1:
                continue  # ignore closing segments that share endpoints
            if segments_intersect(seg_start, seg_end, coords[j], coords[j + 1]):
                return True
    return False


def _cosine_spacing(n: int) -> np.ndarray:
    """cosine-spaced samples between 0 and 1."""
    if n <= 1:
        return np.zeros(1)
    k = np.linspace(0.0, np.pi, n)
    return 0.5 * (1 - np.cos(k))


def resample_airfoil(coords: np.ndarray, panel_points: int) -> np.ndarray:
    """cosine resample with finer resolution near the edges."""
    panel_points = max(int(panel_points), 4)
    upper, lower = split_upper_lower(coords)

    n_upper = max(2, (panel_points + 2) // 2)
    n_lower = max(2, panel_points - n_upper + 2)

    x_upper = _cosine_spacing(n_upper)
    x_lower = _cosine_spacing(n_lower)

    y_upper = np.interp(x_upper, upper[:, 0], upper[:, 1],
                        left=upper[0, 1], right=upper[-1, 1])
    y_lower = np.interp(x_lower, lower[:, 0], lower[:, 1],
                        left=lower[0, 1], right=lower[-1, 1])

    upper_resampled = np.column_stack((x_upper, y_upper))
    lower_resampled = np.column_stack((x_lower, y_lower))

    upper_te2le = upper_resampled[::-1]  # keep upper surface in trailing->leading order
    lower_le2te = lower_resampled[1:]
    return np.vstack((upper_te2le, lower_le2te))
