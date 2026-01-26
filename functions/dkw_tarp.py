# dkw_tarp.py
# Compute and plot DKW bands for PIT/TARP calibration curves.

from typing import Tuple, Optional
import numpy as np

try:
    import jax.numpy as jnp
    JAX = True
except Exception:
    jnp = np  # fall back to NumPy
    JAX = False

def dkw_epsilon(n: int, alpha: float = 0.05) -> float:
    """
    Global (simultaneous) DKW/Massart band half-width for ECDF on [0,1]:
        P( sup_x |F_n(x) - F(x)| > eps ) <= alpha
        eps = sqrt( (1 / (2n)) * ln(2/alpha) )
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")
    return float(np.sqrt(0.5 * np.log(2.0 / alpha) / n))

def ecdf(u: np.ndarray, grid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical CDF (TARP curve) of PIT values u in [0,1].
    Uses searchsorted for O(n log n).
    Returns (grid, F_n(grid)).
    """
    u = np.asarray(u)
    if grid is None:
        grid = np.linspace(0.0, 1.0, 501)
    u_sorted = np.sort(u)
    counts = np.searchsorted(u_sorted, grid, side="right")
    fn = counts / u_sorted.size
    return grid, fn

def dkw_band_on_unit(grid: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    DKW band around the *true* CDF under calibration, which is F(x)=x on [0,1].
    Returns (lower, upper) clipped to [0,1].
    """
    lower = np.clip(grid - eps, 0.0, 1.0)
    upper = np.clip(grid + eps, 0.0, 1.0)
    return lower, upper

# ---------- Optional plotting helper ----------
import matplotlib.pyplot as plt

def plot_tarp_with_dkw(u: np.ndarray,
                       alpha: float = 0.05,
                       grid: Optional[np.ndarray] = None,
                       ax: Optional[plt.Axes] = None):
    """
    Plot the TARP curve (ECDF of PIT), the 45° line, and the DKW simultaneous band.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # TARP (ECDF) of PIT values
    grid, fn = ecdf(u, grid=grid)

    # DKW band and diagonal
    eps = dkw_epsilon(len(u), alpha=alpha)
    lo, hi = dkw_band_on_unit(grid, eps)

    # Fill band, diagonal, and ECDF
    ax.fill_between(grid, lo, hi, alpha=0.2, label=f"DKW {int((1-alpha)*100)}% band")
    ax.plot(grid, grid, linestyle="--", label="Ideal (y = x)")
    ax.step(grid, fn, where="post", label="TARP (ECDF of PIT)")

    ax.set_xlabel("Credibility level (α)")
    ax.set_ylabel("Empirical coverage  Ĉ(α)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.set_title(f"TARP with DKW band (n={len(u)}, α={alpha})")
    return ax

# ---------- Minimal example ----------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    # Example: perfectly calibrated PIT ~ Uniform(0,1)
    u = rng.uniform(0.0, 1.0, size=n)
    plot_tarp_with_dkw(u, alpha=0.05)
    plt.show()
