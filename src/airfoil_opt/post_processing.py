"""
Post-processing analysis and plotting script.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import shutil
from . import config
from .surrogate_model import MultiOutputGP
from .ffd_xfoil_analysis import Xfoil_Analysis, Analysis_Params
from .visualization import set_aiaa_style

pt = 1.0 / 72.27
width = 345 * pt
palette = plt.get_cmap("Dark2").colors

def _read_dat(dat_path: Path) -> np.ndarray:
    return np.loadtxt(dat_path, skiprows=1)

def _read_cp_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, sep=r"\s+", skiprows=3, header=None,
        names=["x", "y", "cp"]
    )

def _run_single_cp_analysis(design_coords: np.ndarray, alpha: float, cp_file: Path) -> pd.DataFrame:
    """Isolates the running of a single XFOIL Cp analysis in a clean environment."""
    temp_dir = Path("./temp_cp_run")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    orig_dirs = (Xfoil_Analysis.DAT_DIR, Xfoil_Analysis.INPUTS_DIR, Xfoil_Analysis.POLAR_DIR)
    Xfoil_Analysis.DAT_DIR = Xfoil_Analysis.INPUTS_DIR = Xfoil_Analysis.POLAR_DIR = temp_dir

    try:
        runner = Xfoil_Analysis("cp_run", {"xy": design_coords}, Analysis_Params())
        runner._write_airfoil_dat()
        runner._write_cp_input(alpha, cp_file)
        if runner._run_xfoil(timeout=30) and cp_file.exists():
            return _read_cp_file(cp_file)
    finally:
        Xfoil_Analysis.DAT_DIR, Xfoil_Analysis.INPUTS_DIR, Xfoil_Analysis.POLAR_DIR = orig_dirs
        shutil.rmtree(temp_dir)
    return pd.DataFrame()

def generate_cp_plots():
    """Generates and plots Cp distributions for the final optimal airfoil."""
    set_aiaa_style()
    dat_files = sorted((config.OPTIMAL_DIR / "dat").glob("opt_*.dat"))
    final_dat_path = dat_files[-1] if dat_files else None
    if not final_dat_path or not final_dat_path.exists():
        print(f"Warning: Final optimal airfoil file not found at {final_dat_path}. Skipping Cp plots.")
        return

    final_coords = _read_dat(final_dat_path)
    best_coords = _load_best_lhs_coords()
    if best_coords.size == 0:
        return
    angles = [0.0, 5.0, 10.0, 15.0]
    
    aero_output_dir = config.IMAGES_DIR / "aero_data"
    aero_output_dir.mkdir(exist_ok=True)

    cols = 2
    rows = 2
    cp_aspect = 7.0 / 10.0  # preserve original 10x7 proportion per subplot
    fig, axes = plt.subplots(rows, cols, figsize=(width*1.2, width), sharex=True, sharey=True)
    for ax, alpha in zip(axes.flatten(), angles):
        cp_final_path = aero_output_dir / f"cp_final_alpha_{alpha:.1f}.txt"
        cp_best_path = aero_output_dir / f"cp_best_alpha_{alpha:.1f}.txt"
        df_final = _run_single_cp_analysis(final_coords, alpha, cp_final_path)
        df_best = _run_single_cp_analysis(best_coords, alpha, cp_best_path)

        _plot_cp_comparison(ax, df_best, df_final)

        ax.set_title(f"Cp at aoa = {alpha:.1f}")
        # only set xlabel on bottom plots
        row = angles.index(alpha) // 2
        ax.set_xlabel("x/c" if row == 1 else "")
        ax.invert_yaxis(); ax.grid(True, alpha=0.3, linestyle="--")

    axes[0,0].set_ylabel("Cp (pressure coefficient)"); axes[1,0].set_ylabel("Cp (pressure coefficient)")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.07, wspace=0.2, hspace=0.26)
    fig.tight_layout()
    fig.savefig(config.IMAGES_DIR / "cp_distributions.png", format="png")
    plt.close(fig)
    print(f"Saved: {config.IMAGES_DIR / 'cp_distributions.png'}")

def generate_surrogate_parity_plots():
    """Generates surrogate vs. XFOIL parity plots to validate model accuracy."""
    if not config.LHS_RESULTS_PATH.exists() or not config.SURROGATE_MODELS_DIR.exists():
        print("Warning: LHS results or surrogate models not found. Skipping parity plots.")
        return
    set_aiaa_style()
    with config.LHS_RESULTS_PATH.open("rb") as f:
        lhs_results = pickle.load(f)
    gp = MultiOutputGP.load(config.SURROGATE_MODELS_DIR)
    targets = ["CL_max", "CD", "CM"]
    
    cols = 3
    rows = 1
    fig, axes = plt.subplots(rows, cols, figsize=(width*1.6, width*0.5))

    for ax, key in zip(axes, targets):
        pts_train, pts_val = [], []
        for i, rec in enumerate(lhs_results):
            if not rec.get("xfoil", {}).get("converged"):
                continue
            
            pred = gp.predict(rec["xy"], return_std=False).get(key)
            true = rec["xfoil"].get(key)
            
            # Add this point to either training or validation set
            (pts_val if i % 4 == 0 else pts_train).append((true, pred))

        ax.scatter(*zip(*pts_train), color=palette[0], alpha=0.7, s=30, label="Train", edgecolors="white", linewidths=0.3)
        ax.scatter(*zip(*pts_val), color=palette[1], alpha=0.8, s=40, label="Validation", marker="D", edgecolors="white", linewidths=0.3)
        
        all_vals = [t for t, _ in pts_train + pts_val]
        lo, hi = min(all_vals), max(all_vals)
        pad = 0.05 * (hi - lo + 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="k", lw=1.0, ls="--", label="Ideal")
        
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        ax.set_title(f"{key} Parity")
        ax.set_xlabel("XFOIL True Value")
        ax.set_ylabel("Surrogate Predicted Value")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.1, wspace=0.34, hspace=0.0)
    fig.savefig(config.IMAGES_DIR / "surrogate_parity.png", format="png")
    plt.close(fig)
    print(f"Saved: {config.IMAGES_DIR / 'surrogate_parity.png'}")

def run_all_post_processing():
    generate_cp_plots()
    generate_surrogate_parity_plots()

def _plot_cp_comparison(ax, df_best: pd.DataFrame, df_final: pd.DataFrame):
    best_upper, best_lower = _split_cp_surfaces(df_best)
    final_upper, final_lower = _split_cp_surfaces(df_final)

    series = [
        ("best lhs upper", best_upper, palette[0], "-", "o", False, 0.8, 1.0),
        ("best lhs lower", best_lower, palette[0], "-", "v", True, 0.8, 1.0),
        ("final upper", final_upper, palette[1], "--", "o", False, 1.0, 0.8),
        ("final lower", final_lower, palette[1], "--", "v", True, 1.0, 0.8),
    ]

    handles = []
    labels = []
    for label, subset, color, linestyle, marker, filled, lw, alpha in series:
        if subset.empty:
            continue
        markevery = max(1, len(subset) // 30)
        handle, = ax.plot(
            subset["x"], subset["cp"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markerfacecolor=color if filled else "none",
            markeredgecolor=color,
            markevery=markevery,
            linewidth=lw,
            alpha=alpha
        )
        handles.append(handle)
        labels.append(label)

    if handles:
        ax.legend(handles, labels, frameon=False)

def _split_cp_surfaces(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    idx = int(df["x"].idxmin())
    upper = df.iloc[: idx + 1].copy()
    lower = df.iloc[idx:].copy()
    return upper.sort_values("x"), lower.sort_values("x")

def _load_best_lhs_coords() -> np.ndarray:
    if not config.LHS_RESULTS_PATH.exists():
        return np.empty((0, 0))
    with config.LHS_RESULTS_PATH.open("rb") as handle:
        lhs_results = pickle.load(handle)

    best_coords = None
    best_obj = np.inf
    for record in lhs_results:
        name = str(record.get("name", ""))
        coords = record.get("xy")
        obj = record.get("J")
        if coords is None or obj is None or not name.startswith("lhs"):
            continue
        if obj < best_obj:
            best_obj = obj
            best_coords = np.asarray(coords, dtype=float)

    return best_coords if best_coords is not None else np.empty((0, 0))
