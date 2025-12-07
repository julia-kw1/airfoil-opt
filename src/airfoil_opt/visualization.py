"""
unified visualization and diagnostic plotting utilities.
"""
import numpy as np
import pandas as pd
import matplotlib
import pickle
from pandas.plotting import parallel_coordinates

matplotlib.use("agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List
from matplotlib.ticker import MaxNLocator
from . import config
from .ffd_geometry import FFDBox2D
from .seed_generation import generate_seed_airfoils
from .geometry_utils import extract_camberline

pt = 1./72.27
width = 345*pt

def set_aiaa_style():
    # Use LaTeX for all text
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        # match typical LaTeX body size
        "font.size": 10,
        "axes.labelsize": 8,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 9,
        "lines.markersize": 3,

        # Tight layout
        "figure.autolayout": True,

        # Savefig defaults
        "savefig.dpi": 200
    })

    # Optional: slightly nicer grid
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.4

    # Return plt just for convenience
    return plt

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.get_cmap("Dark2").colors)
palette = plt.get_cmap("Dark2").colors

def create_all_plots(
    seed_airfoils: List[Dict],
    best_lhs: Dict,
    final_design: Dict,
    all_lhs_results: List[Dict],
    convergence_diagnostics: List[Dict]
):    
    """generate all standard diagnostic plots from an optimization run."""
    set_aiaa_style()
    print("\n--- generating plots ---")
    plot_geometry_comparison(seed_airfoils, best_lhs, final_design)
    plot_lhs_geometry_gallery(seed_airfoils, all_lhs_results)
    plot_polar_data(best_lhs.get("name"), final_design.get("name"))
    plot_optimization_convergence(all_lhs_results, convergence_diagnostics)
    plot_lhs_performance_distribution(all_lhs_results, best_lhs, final_design)
    plot_lhs_parallel_coordinates(all_lhs_results)
    plot_seed_airfoils(seed_airfoils)
    print("all standard plots saved successfully.")

def plot_geometry_comparison(seed_airfoils, best_lhs, final):
    if not best_lhs or not final:
        print("skipping geometry plot; missing best_lhs or final design.")
        return
        
    seed_idx_best = best_lhs["params"]["seed_idx"]
    seed_idx_final = final.get("params", {}).get("seed_idx", seed_idx_best)
    base_seed_best = seed_airfoils[seed_idx_best]["xy"]
    base_seed_final = seed_airfoils[seed_idx_final]["xy"]

    best_coords = best_lhs["xy"]
    final_coords = final["xy"]
    camber_best = _camberline_from_record(best_lhs, best_coords)
    camber_final = _camberline_from_record(final, final_coords)

    ffd_best = FFDBox2D.from_airfoil(base_seed_best, pad=(config.FFD_PAD_X, config.FFD_PAD_Y), grid=(config.FFD_NX, config.FFD_NY))
    dP_best = best_lhs.get("dP")
    cp_best = ffd_best.control_points + (np.asarray(dP_best, dtype=float) if dP_best is not None else np.zeros_like(ffd_best.control_points))
    ffd_final = FFDBox2D.from_airfoil(base_seed_final, pad=(config.FFD_PAD_X, config.FFD_PAD_Y), grid=(config.FFD_NX, config.FFD_NY))
    dP_final = final.get("dP")
    cp_final = ffd_final.control_points + (np.asarray(dP_final, dtype=float) if dP_final is not None else np.zeros_like(ffd_final.control_points))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, width/4), sharex=True, sharey=True)
    _plot_airfoil_ffd(
        ax1, best_coords, cp_best,
        f"Best LHS - {best_lhs.get('name', 'unknown')}",
        camber=camber_best
    )
    _plot_airfoil_ffd(
        ax2, final_coords, cp_final,
        "Final Optimal Geometry",
        camber=camber_final
    )
    ax2.set_ylabel("")

    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.98, top=0.8, bottom=0.2, wspace=0.0, hspace=0.10)
    fig.savefig(config.IMAGES_DIR / "final_geometry_comparison.png", format="png")
    plt.close(fig)
    print(f"saved: {config.IMAGES_DIR / 'final_geometry_comparison.png'}")

def plot_lhs_geometry_gallery(seed_airfoils: List[Dict], lhs_results: List[Dict], n_samples: int = 6):
    lhs_records = [r for r in lhs_results if str(r.get("name", "")).startswith("lhs") and r.get("xy") is not None]
    if not lhs_records:
        return

    sample_count = min(n_samples, len(lhs_records))
    rng = np.random.default_rng(0)
    indices = rng.choice(len(lhs_records), size=sample_count, replace=False)

    fig, axes = plt.subplots(3, 2, figsize=(width*1.5, width), sharex=True, sharey=True)
    axes = axes.flatten()
    total_cols = 2

    for ax, idx in zip(axes, indices):
        record = lhs_records[int(idx)]
        coords = np.asarray(record["xy"])
        camber = _camberline_from_record(record, coords)
        seed_idx = record.get("params", {}).get("seed_idx", 0)
        base_seed = seed_airfoils[seed_idx]["xy"] if 0 <= seed_idx < len(seed_airfoils) else coords
        ffd = FFDBox2D.from_airfoil(base_seed, pad=(config.FFD_PAD_X, config.FFD_PAD_Y), grid=(config.FFD_NX, config.FFD_NY))
        dP = record.get("dP")
        deltas = np.asarray(dP, dtype=float) if dP is not None else np.zeros_like(ffd.control_points)
        control = ffd.control_points + deltas
        _plot_airfoil_ffd(ax, coords, control, "", camber=camber)

        idx_pos = list(axes).index(ax)
        row, col = divmod(idx_pos, total_cols)
        n_rows = int(np.ceil(len(axes) / total_cols))
        ax.set_ylabel("y/c" if col == 0 else "")
        ax.set_xlabel("x/c" if row == n_rows - 1 else "")
        if camber.size > 0:
            airfoil_line = ax.lines[0]
            camber_line = ax.lines[1]
            ax.legend(
                [airfoil_line, camber_line],
                [record.get("name", "lhs")],
                frameon=False, loc="upper right",
                handlelength=1.8, borderpad=0.2, labelspacing=0.2,
            )
        ax.set_yticks(np.linspace(-0.2, 0.2, 5))
        ax.set_xticks(np.linspace(0, 1, 5))
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.25, 0.25)
        ax.grid(True, alpha=0.5, linestyle="--")

    for ax in axes[sample_count:]:
        ax.axis("off")

    fig.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.09, wspace=0.00, hspace=0.00)
    fig.savefig(config.IMAGES_DIR / "lhs_geometry_samples.png", format="png")
    plt.close(fig)
    print(f"saved: {config.IMAGES_DIR / 'lhs_geometry_samples.png'}")

def plot_polar_data(best_lhs_name, final_name):
    if not best_lhs_name or not final_name:
        print("skipping polar plot; design names are missing.")
        return

    df_best = _read_polar(config.POLAR_DIR / f"{best_lhs_name}_polar.txt")
    df_final = _read_polar(config.OPTIMAL_DIR / "opt_final_polar.txt")

    if df_best.empty or df_final.empty:
        print("skipping polar plot; one or both polar files could not be read.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(width*1.2, width))
    axs = axes.flatten()
    keys = [("alpha", "cl"), ("alpha", "cd"), ("alpha", "cm"), ("cd", "cl")]
    titles = [
        "Lift vs. Angle of Attack",
        "Drag vs. Angle of Attack",
        "Moment vs. Angle of Attack",
        "Drag Polar"
    ]
    axis_labels = {
        "alpha": "Angle of Attack",
        "cl": "C_L",
        "cd": "C_D",
        "cm": "C_M",
    }

    for ax, (x, y), title in zip(axs, keys, titles):
        ax.plot(
            df_best[x], df_best[y],
            label="Best LHS",
            color=palette[0],
            lw=1.0,
            ls="--",
            marker="o",
            markerfacecolor="none",
            markeredgecolor=palette[0],
        )
        ax.plot(
            df_final[x], df_final[y],
            label="Final Optimal",
            color=palette[1],
            lw=1.0,
            marker="v",
            markerfacecolor=palette[1],
            markeredgecolor=palette[1],
        )
        ax.set_title(title)
        ax.set_xlabel(axis_labels.get(x, x.title()))
        ax.set_ylabel(axis_labels.get(y, y.title()))
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, prune=None))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7, prune=None))
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(left=0.11, right=0.99, top=0.96, bottom=0.07, wspace=0.21, hspace=0.28)
    fig.savefig(config.IMAGES_DIR / "final_polar_comparison.png", format="png")
    plt.close(fig)
    print(f"saved: {config.IMAGES_DIR / 'final_polar_comparison.png'}")

def plot_optimization_convergence(all_results: List[Dict], diagnostics_data: List[Dict] = None):
    set_aiaa_style()
    if not config.OPT_HISTORY_PATH.exists():
        print(f"skipping ego convergence plot; {config.OPT_HISTORY_PATH} not found.")
        return
    with config.OPT_HISTORY_PATH.open("rb") as f: history = pickle.load(f)
    df = pd.DataFrame(history)
    diag_df = _prepare_convergence_dataframe(diagnostics_data or [])

    running = []
    best_so_far = np.inf
    for idx, res in enumerate(all_results):
        J_val = res.get("J")
        if J_val is None: 
            continue
        best_so_far = min(best_so_far, J_val)
        phase = "lhs" if str(res.get("name", "")).startswith("lhs") else "active"
        running.append({"eval": idx + 1, "J": J_val, "best": best_so_far, "phase": phase})
    run_df = pd.DataFrame(running)

    fig, axes = plt.subplots(2, 2, figsize=(width*1.4, width*0.8))
    axes = axes.flatten()

    # show xlables in scientific notation for better readability
    axes[0].plot(run_df["eval"], run_df["best"], lw=1.0, label="Running Best J", color=palette[0])
    axes[0].scatter(run_df["eval"], run_df["J"], alpha=0.7, label="Sampled J", color=palette[1])
    axes[0].set_title("Running Best Objective (All Samples)")
    axes[0].set_xlabel("Evaluation \\#")
    axes[0].set_ylabel("Objective J")
    axes[0].tick_params(axis='y', rotation=45)
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].legend(frameon=False, loc="center right")


    axes[1].plot(df["iteration"], df["J_pred"], lw=1.0, label="Predicted J")
    axes[1].set_title("EGO Predicted Objective per SLSQP Iteration")
    axes[1].set_xlabel("Optimizer Iteration")
    axes[1].set_ylabel("Predicted J")
    axes[1].tick_params(axis='y', rotation=45)
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].legend(frameon=False)

    if "sigma_pred" in df.columns:
        axes[2].plot(df["iteration"], df["sigma_pred"], lw=1.0, label="Exploration Sigma")
        axes[2].set_ylabel("Scaled Predictive Sigma")
    if "EI" in df.columns:
        axes[2].plot(df["iteration"], df["EI"], lw=1.0, label="Expected Improvement")
    axes[2].set_title("Exploration vs. Improvement Signals")
    axes[2].set_xlabel("Optimizer Iteration")
    axes[2].tick_params(axis='y', rotation=45)
    axes[2].grid(True, alpha=0.3, linestyle="--")
    axes[2].legend(frameon=False)


    if "EI" in df.columns:
        axes[3].bar(df["iteration"], df["EI"], width=0.6, color=palette[0])
        axes[3].set_title("Expected Improvement (Bar View)")
        axes[3].set_xlabel("Optimizer Iteration")
        axes[3].set_ylabel("Expected Improvement")
        axes[3].tick_params(axis='y', rotation=45)
        axes[3].grid(True, alpha=0.25, linestyle="--", axis="y")
    else:
        axes[3].axis("off")

    fig.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.1, wspace=0.32, hspace=0.4)
    fig.savefig(config.IMAGES_DIR / "optimization_convergence.png", format="png")
    plt.close(fig)
    print(f"saved: {config.IMAGES_DIR / 'optimization_convergence.png'}")

    if diag_df is not None:
        fig, ax = plt.subplots(1, 1, figsize=(width*0.9, width*0.6))
        _plot_convergence_panel(ax, diag_df, "Optimality Gap and Max EI")
        fig.tight_layout()
        fig.savefig(config.IMAGES_DIR / "convergence_diagnostics.png", format="png")
        plt.close(fig)
        print(f"saved: {config.IMAGES_DIR / 'convergence_diagnostics.png'}")
    else:
        print("skipping convergence diagnostics plot; no convergence data available.")
    

def plot_lhs_performance_distribution(lhs_results, best_lhs: Dict = None, final_design: Dict = None):
    converged_xfoil = [r['xfoil'] for r in lhs_results if r.get("xfoil", {}).get("converged")]
    if not converged_xfoil:
        print("skipping lhs distribution plot; no converged xfoil results found.")
        return
    df = pd.DataFrame(converged_xfoil)
    
    best_xfoil = best_lhs.get("xfoil", {}) if best_lhs else {}
    final_xfoil = final_design.get("xfoil", {}) if final_design else {}

    fig, axes = plt.subplots(2, 2, figsize=(width*1.2, 0.84*width))
    axes = axes.flatten()

    axes[0].hist(df['CL_max'], bins=15, alpha=0.8)
    axes[0].set_title("CL_max Distribution")
    axes[0].set_xlabel("CL_max")
    axes[0].set_ylabel("Count")
    _mark_hist_lines(axes[0], best_xfoil.get("CL_max"), final_xfoil.get("CL_max"), palette)

    axes[1].hist(df['CD'], bins=15, alpha=0.8)
    axes[1].set_title("CD at CL_max Distribution")
    axes[1].set_xlabel("CD")
    _mark_hist_lines(axes[1], best_xfoil.get("CD"), final_xfoil.get("CD"), palette)

    axes[2].hist(df['CM'], bins=15, alpha=0.8)
    axes[2].set_title("CM at CL_max Distribution")
    axes[2].set_xlabel("CM")
    _mark_hist_lines(axes[2], best_xfoil.get("CM"), final_xfoil.get("CM"), palette)

    scatter = axes[3].scatter(df['CD'], df['CL_max'], c=df['CM'], cmap="viridis", edgecolors="white", alpha=0.8, s=34)
    axes[3].set_title("Drag Polar Colored by CM")
    axes[3].set_xlabel("CD")
    axes[3].set_ylabel("CL_max")

    if best_xfoil:
        axes[3].scatter(best_xfoil.get("CD"), best_xfoil.get("CL_max"), color=palette[0], s=30, marker="d", label="Best LHS", edgecolors="k", zorder=5)
    if final_xfoil:
        axes[3].scatter(final_xfoil.get("CD"), final_xfoil.get("CL_max"), color=palette[1], s=30, marker="*", label="Final Optimal", edgecolors="k", zorder=5)
    if best_xfoil or final_xfoil:
        axes[3].legend(frameon=False, loc="lower right")

    fig.colorbar(scatter, ax=axes[3], label="CM at CL_max")

    for ax in axes: ax.grid(True, alpha=0., linestyle="--")
    fig.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.09, wspace=0.21, hspace=0.35)
    fig.savefig(config.IMAGES_DIR / "lhs_performance_distribution.png", format="png")
    plt.close(fig)
    print(f"saved: {config.IMAGES_DIR / 'lhs_performance_distribution.png'}")

def plot_lhs_parallel_coordinates(lhs_results: List[Dict], max_vars: int = 10):
    matrix = _lhs_design_matrix(lhs_results)
    if matrix.size == 0 or matrix.shape[1] < 2:
        return

    scaled = _scale_top_variance(matrix, max_vars)
    if scaled.shape[1] < 2:
        return

    save_path = config.IMAGES_DIR / "lhs_parallel_coordinates.png"

    columns = [f"DOF {i + 1}" for i in range(scaled.shape[1])]
    frame = pd.DataFrame(scaled, columns=columns)
    frame.insert(0, "group", "LHS")

    plt.figure(figsize=(width, (width*1.2)/2.5))
    parallel_coordinates(frame, "group", color=[palette[0]], linewidth=1.0, alpha=0.85)
    plt.ylabel("Scaled Design Variable Magnitude")
    plt.yticks(np.linspace(0.0, 1.0, 5))
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(False)
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(save_path, dpi=350, bbox_inches="tight")
    plt.close()

def plot_seed_airfoils(seed_airfoils: List[Dict]):
    if not seed_airfoils:
        return

    n_airfoils = len(seed_airfoils)
    fig, axes = plt.subplots(n_airfoils, 1, figsize=(width*0.8, (width*0.8)*1.39), sharex=True)
    if n_airfoils == 1:
        axes = [axes]

    for idx, (seed_dict, ax) in enumerate(zip(seed_airfoils, axes)):
        coords = seed_dict["xy"]
        airfoil_line, = ax.plot(coords[:, 0], coords[:, 1], color="k", lw=1.0, label=seed_dict["name"])

        camber = seed_dict.get("camberline")
        if camber is None or camber.size == 0:
            camber = extract_camberline(coords)

        camber_line = None
        if camber.size > 0:
            camber_line, = ax.plot(camber[:, 0], camber[:, 1], color=palette[0], lw=0.6, ls="--")

        ax.set_aspect("equal")
        ax.grid(False)
        if idx == n_airfoils - 1:
            ax.set_xlabel("x/c")
            ax.set_xlim(-0.05, 1.05)
            ax.set_xticks(np.linspace(0, 1, 5))
        else:
            ax.set_xticklabels([])
        ax.set_ylim(-0.14, 0.14)
        ax.set_yticks(np.linspace(-0.1, 0.1, 3))
        ax.set_ylabel("y/c")

        handles = [airfoil_line]
        labels = [seed_dict["name"]]
        if camber_line is not None:
            handles.append(camber_line)
        ax.legend(handles, labels, frameon=False, loc="upper right")
        ax.grid(True, alpha=0.25, linestyle="--")

    fig.subplots_adjust(left=0.05, right=0.98, top=1.0, bottom=0.05, hspace=0.00, wspace=0.00)
    fig.savefig(config.IMAGES_DIR / "seed_airfoils.png", format="png")
    plt.close(fig)
    print(f"saved: {config.IMAGES_DIR / 'seed_airfoils.png'}")

def _plot_airfoil_ffd(ax, coords, control_points, title, camber: np.ndarray = None):
    ax.plot(coords[:, 0], coords[:, 1], color="k", lw=1.0, zorder=4)
    if camber is not None and camber.size > 0:
        ax.plot(camber[:, 0], camber[:, 1], color=palette[0], lw=0.6, ls="--", zorder=5)

    cp_color = palette[1]
    cp_alpha = 0.6
    nx, ny, _ = control_points.shape
    flat_cp = control_points.reshape(-1, 2)
    ax.scatter(flat_cp[:, 0], flat_cp[:, 1], color=cp_color, alpha=cp_alpha, zorder=3, s=1.5)
    for i in range(nx):
        ax.plot(control_points[i, :, 0], control_points[i, :, 1], color=cp_color, lw=0.8, alpha=cp_alpha, ls="--")
    for j in range(ny):
        ax.plot(control_points[:, j, 0], control_points[:, j, 1], color=cp_color, lw=0.8, alpha=cp_alpha, ls="--")

    ax.set_title(title)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")

def _read_polar(polar_path: Path) -> pd.DataFrame:
    if not polar_path.exists(): return pd.DataFrame()
    try:
        df = pd.read_csv(
            polar_path, sep=r'\s+', skiprows=12, header=None,
            names=['alpha', 'cl', 'CD', 'CDp', 'cm', 'top_xtr', 'bot_xtr'],
            on_bad_lines='skip', dtype=float,
        )
        # column names in pandas are case-sensitive, ensure lowercase
        df.columns = [c.lower() for c in df.columns]
        return df.sort_values("alpha").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def _mark_hist_lines(ax, best_val, final_val, palette):
    if best_val is None and final_val is None: return
    handles = []
    labels = []
    if best_val is not None:
        h_best = ax.axvline(best_val, color=palette[0], lw=1.0, ls="--")
        handles.append(h_best); labels.append("Best LHS")
    if final_val is not None:
        h_final = ax.axvline(final_val, color=palette[1], lw=1.0, ls="-.")
        handles.append(h_final); labels.append("Final Optimal")
    ax.legend(handles, labels, frameon=False)

def _camberline_from_record(record: Dict, coords: np.ndarray) -> np.ndarray:
    camber = None
    if isinstance(record, dict):
        camber = record.get("camberline")
    if camber is None or len(np.atleast_1d(camber)) == 0:
        return extract_camberline(coords)
    camber_arr = np.asarray(camber, dtype=float)
    if camber_arr.ndim != 2 or camber_arr.shape[1] != 2:
        return extract_camberline(coords)
    return camber_arr

def _prepare_convergence_dataframe(diagnostics_data: List[Dict]):
    if not diagnostics_data:
        return None
    df = pd.DataFrame(diagnostics_data).dropna()
    return df if not df.empty else None

def _plot_convergence_panel(ax, df: pd.DataFrame, title: str):
    color1 = palette[0]
    color2 = palette[1]

    ax.set_xlabel("Active Learning Iteration")
    ax.set_ylabel("Optimality Gap", color=color1)
    ax.plot(df["iteration"], df["optimality_gap"], marker="o", linestyle="-", color=color1, label="Optimality Gap")
    ax.tick_params(axis="y", labelcolor=color1)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, axis="y")
    ax.axhline(y=config.ACTIVE_J_TOL, color=color1, linestyle=":", linewidth=1, label=f"Gap Tolerance ({config.ACTIVE_J_TOL})")
    
    ax2 = ax.twinx()
    ax2.set_ylabel("Max Expected Improvement (Regret)", color=color2)
    ax2.plot(df["iteration"], df["max_expected_improvement"], marker="s", linestyle="--", color=color2, label="Max EI (Regret)")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_yscale("log")
    ax2.axhline(y=config.EI_REGRET_TOLERANCE, color=color2, linestyle=":", linewidth=1, label=f"EI Tolerance ({config.EI_REGRET_TOLERANCE})")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center left", frameon=False)
    ax.set_title(title)

def _lhs_design_matrix(lhs_results: List[Dict]) -> np.ndarray:
    rows = []
    for entry in lhs_results:
        name = str(entry.get("name", ""))
        if not name.startswith("lhs"):
            continue
        dP = entry.get("dP")
        if dP is None:
            continue
        vec = np.asarray(dP, dtype=float).ravel()
        if vec.size == 0 or not np.all(np.isfinite(vec)):
            continue
        rows.append(vec)

    if not rows:
        return np.empty((0, 0))

    matrix = np.vstack(rows)
    variance = matrix.var(axis=0)
    mask = variance > 1e-12
    return matrix[:, mask] if np.any(mask) else np.empty((0, 0))

def _scale_top_variance(matrix: np.ndarray, max_vars: int) -> np.ndarray:
    if matrix.size == 0:
        return np.empty((matrix.shape[0], 0))

    n_cols = matrix.shape[1]
    n_vars = max(2, min(max_vars, n_cols))
    variance = matrix.var(axis=0)
    idx = np.argsort(variance)[::-1][:n_vars]

    subset = matrix[:, idx]
    col_min = subset.min(axis=0)
    span = subset.max(axis=0) - col_min
    span[span < 1e-12] = 1.0
    return (subset - col_min) / span
