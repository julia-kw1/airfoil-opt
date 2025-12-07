"""Objective function and weights for optimization."""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from . import config
from .geometry_utils import thickness_ratio, has_self_intersection

@dataclass
class Weights:
    """
    Objective function weights and targets.
    This class dynamically loads all its values from the main config file.
    """
    def __init__(self):
        for key, value in config.WEIGHTS.items():
            setattr(self, key, value)

def score_design(xfoil_result: Dict[str, Any], coords: np.ndarray, weights: Weights) -> float:
    """
    Computes the objective function value (to be minimized) based on XFOIL
    results and geometry, using the provided weights.
    """
    cl = float(xfoil_result.get('CL_max', np.nan))
    alpha = float(xfoil_result.get('alpha_CL_max', np.nan))
    cd = float(xfoil_result.get('CD', np.nan))
    cm = float(xfoil_result.get('CM', np.nan))
    cd0 = float(xfoil_result.get('CD_alpha0', np.nan))
    cm0 = float(xfoil_result.get('CM_alpha0', np.nan))

    is_prediction = 'converged' not in xfoil_result

    if not is_prediction and not xfoil_result['converged']:
        return 1e5
    
    if np.isfinite(cl) and cl > 0.3:
        base_penalty = -cl * weights.w_cl
    elif is_prediction:
        base_penalty = -cl * weights.w_cl if np.isfinite(cl) else 0.0
    else:
        base_penalty = 1e3

    # stall angle penalty (and small bonus for exceeding target)
    if np.isfinite(alpha):
        delta = max(0.0, weights.alpha_target - alpha)
        alpha_penalty = weights.w_alpha * delta**2
    else:
        alpha_penalty = weights.w_alpha * weights.alpha_target**2

    # drag penalties (at CL_max and at alpha=0)
    cd_penalty = weights.w_cd * (cd if np.isfinite(cd) and cd > 0 else 0.1)
    cd0_penalty = weights.w_cd0 * max(0.0, cd0 - weights.cd0_target)**2 if np.isfinite(cd0) else weights.w_cd0 * 0.05

    # geometric constraint penalties
    thickness = thickness_ratio(coords)
    t_upper_penalty = weights.w_t_upper * max(0, thickness - 0.24)**2 if np.isfinite(thickness) else 0.0
    t_lower_penalty = weights.w_t_lower * max(0, 0.12 - thickness)**2 if np.isfinite(thickness) else 0.0
    overlap_penalty = weights.w_overlap if has_self_intersection(coords) else 0.0

    # pitching moment penalties (at CL_max and at alpha=0)
    cm_val = cm if np.isfinite(cm) else -0.15
    cm_delta = abs(cm_val - weights.cm_target)
    cm_excess = cm_delta - weights.cm_tolerance
    cm_penalty = weights.w_cm * (cm_excess**2) if cm_excess > 0 else 0.0

    cm0_val = cm0 if np.isfinite(cm0) else -0.15
    cm0_delta = abs(cm0_val - weights.cm0_target)
    cm0_excess = cm0_delta - weights.cm_tolerance
    cm0_penalty = weights.w_cm0 * (cm0_excess**2) if cm0_excess > 0 else 0.0
    
    # boundary layer transition penalty
    top_xtr = float(xfoil_result.get('Top_Xtr', np.nan))
    bot_xtr = float(xfoil_result.get('Bot_Xtr', np.nan))
    trans_penalty = weights.w_transition * (
        max(0, weights.xtr_min - (top_xtr if np.isfinite(top_xtr) else 0)) +
        max(0, weights.xtr_min - (bot_xtr if np.isfinite(bot_xtr) else 0))
    )
    """
    # 7. Flow detachment penalty
    detach_penalty = 0.0
    if np.isfinite(top_xtr):
        detach_amt = max(0.0, weights.xtr_detach_min - top_xtr)
        alpha_gap = max(0.0, alpha_target - (alpha if np.isfinite(alpha) else 0.0))
        detach_penalty = weights.w_detach * (detach_amt * (1.0 + alpha_gap / max(1.0, alpha_target)))**2
    """
    total = sum([
        base_penalty, alpha_penalty, cd_penalty, cd0_penalty,
        cm_penalty, cm0_penalty, t_upper_penalty, t_lower_penalty,
        overlap_penalty, trans_penalty #W detach_penalty
    ])
    return total
