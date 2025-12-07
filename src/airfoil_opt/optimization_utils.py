"""optimization helper functions."""
import numpy as np
from typing import Tuple, Callable

from .geometry_utils import thickness_ratio, has_self_intersection


def unpack_dofs(mask: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """rebuild the full control delta array."""
    control_deltas = np.zeros(mask.shape, dtype=float)
    control_deltas[mask] = vec
    return control_deltas


def thickness_constraint_min(coords: np.ndarray, t_min: float = 0.12) -> float:
    """enforce a minimum thickness."""
    t = thickness_ratio(coords)
    return (t if np.isfinite(t) else -1.0) - t_min


def thickness_constraint_max(coords: np.ndarray, t_max: float = 0.35) -> float:
    """enforce a maximum thickness."""
    t = thickness_ratio(coords)
    return t_max - (t if np.isfinite(t) else 0.40)


def no_intersection_constraint(coords: np.ndarray) -> float:
    """positive value when the airfoil is non-intersecting."""
    return 1.0 if not has_self_intersection(coords) else -1.0


def create_constraints(vec_to_coords: Callable):
    """build callable inequality constraints for SLSQP."""
    def constraint_t_min(vec):
        coords, _ = vec_to_coords(vec)
        return thickness_constraint_min(coords)

    def constraint_t_max(vec):
        coords, _ = vec_to_coords(vec)
        return thickness_constraint_max(coords)

    def constraint_intersect(vec):
        coords, _ = vec_to_coords(vec)
        return no_intersection_constraint(coords)

    return (
        {'type': 'ineq', 'fun': constraint_t_min},
        {'type': 'ineq', 'fun': constraint_t_max},
        {'type': 'ineq', 'fun': constraint_intersect}
    )


def _setup_bounds(mask, x_bound, y_bound):
    """pair symmetric bounds for the selected degrees of freedom."""
    limit = np.zeros(mask.shape, dtype=float)
    limit[:, :, 0] = x_bound
    limit[:, :, 1] = y_bound
    selected_limits = limit[mask]
    return [(-float(val), float(val)) for val in selected_limits]
