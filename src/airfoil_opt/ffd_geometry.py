import numpy as np
from scipy.special import comb
from dataclasses import dataclass
from typing import Tuple, Optional

def batch_bernstein(n: int, u: np.ndarray) -> np.ndarray:
    """return all bernstein bases for degree n at u."""
    u = np.asarray(u, dtype=float).reshape(-1, 1)
    powers = np.arange(n + 1)
    coeffs = comb(n, powers)
    return coeffs * np.power(u, powers) * np.power(1.0 - u, n - powers)

@dataclass
class FFDBox2D:
    """planar bezier lattice for free-form deformation."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int
    control_points: np.ndarray

    @staticmethod
    def from_airfoil(coords: np.ndarray,
                     pad: Tuple[float, float] = (0.05, 0.2),
                     grid: Tuple[int, int] = (5, 5)) -> "FFDBox2D":
        """wrap an airfoil with a padded lattice."""
        x_min = -pad[0]
        x_max = 1.0 + pad[0]
        y_min = float(np.min(coords[:, 1]) - pad[1])
        y_max = float(np.max(coords[:, 1]) + pad[1])

        nx, ny = grid
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid, indexing="ij")
        control_points = np.stack((grid_x, grid_y), axis=-1)
        return FFDBox2D(x_min, x_max, y_min, y_max, nx, ny, control_points)

    def deform(self, coords: np.ndarray, control_deltas: Optional[np.ndarray] = None) -> np.ndarray:
        """apply lattice displacements to coordinates."""
        if control_deltas is None:
            active_points = self.control_points
        else:
            active_points = self.control_points + control_deltas
            active_points = active_points.copy()
            active_points[[0, -1], :, :] = self.control_points[[0, -1], :, :]

        span_x = self.x_max - self.x_min
        span_y = self.y_max - self.y_min
        s = np.clip((coords[:, 0] - self.x_min) / span_x, 0.0, 1.0)
        t = np.clip((coords[:, 1] - self.y_min) / span_y, 0.0, 1.0)

        basis_s = batch_bernstein(self.nx - 1, s)
        basis_t = batch_bernstein(self.ny - 1, t)

        weights = np.einsum("ni,nj->nij", basis_s, basis_t)
        deformed = np.einsum("nij,ijk->nk", weights, active_points)

        return deformed
