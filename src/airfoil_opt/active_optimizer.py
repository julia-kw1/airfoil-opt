"""
main optimization pipeline with active-learning loop.
"""
import pickle
import time
import shutil
from pathlib import Path
from scipy.stats import norm
from copy import deepcopy
from . import config
from .seed_generation import generate_seed_airfoils
from .surrogate_model import train_surrogate_models
from .gradient_refiner import run_gradient_refinement
from .visualization import create_all_plots
from .ffd_xfoil_analysis import Xfoil_Analysis, Analysis_Params
from .objective_function import score_design, Weights
from .geometry_utils import thickness_ratio
from .lhs_engine import run_lhs_sampling

def load_pickle_required(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"required file not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)

class ActiveAirfoilOptimizer:
    """orchestrates the ego optimization pipeline."""
    def __init__(self):
        self.params = Analysis_Params()
        self.seed_airfoils = []
        self.lhs_results = []
        self.best_lhs = {}
        self.initial_best_lhs = {}
        self.final_design = {}
        self.convergence_diagnostics = []

    def run(self):
        """execute the full optimization pipeline."""
        self._setup_directories()
        self._pipeline_start_time = time.perf_counter()
        print("\n" + "=" * 64 + "\nactive airfoil optimization pipeline\n" + "=" * 64)

        self._step_1_load_seeds()
        self._step_2_run_lhs()
        self._step_3_active_loop()
        self._step_4_visualize()

        total_time = time.perf_counter() - self._pipeline_start_time
        print(f"\npipeline completed in {total_time/60:.1f} min")

    def _setup_directories(self):
        config.IMAGES_DIR.mkdir(exist_ok=True)
        config.DATA_DIR.mkdir(exist_ok=True)
        if config.OPTIMAL_DIR.exists():
            shutil.rmtree(config.OPTIMAL_DIR)
        (config.OPTIMAL_DIR / "dat").mkdir(parents=True)
        (config.OPTIMAL_DIR / "input").mkdir(parents=True)
        (config.OPTIMAL_DIR / "polar").mkdir(parents=True)

    def _step_1_load_seeds(self):
        print("\n=== step 1: loading seed airfoils ===")
        self.seed_airfoils = generate_seed_airfoils()
        print(f"loaded {len(self.seed_airfoils)} seed airfoil(s).")

    def _step_2_run_lhs(self):
        print("\n=== step 2: latin hypercube sampling ===")
        if config.REUSE_LHS_RESULTS and config.LHS_RESULTS_PATH.exists():
            print(f"loading cached lhs from {config.LHS_RESULTS_PATH}")
            self.lhs_results = load_pickle_required(config.LHS_RESULTS_PATH)
        else:
            Xfoil_Analysis.init_dirs()
            self.lhs_results = run_lhs_sampling(self.seed_airfoils, self.params)
            with config.LHS_RESULTS_PATH.open("wb") as f: pickle.dump(self.lhs_results, f)
        
        self.best_lhs = self._get_best_record()
        self.initial_best_lhs = deepcopy(self.best_lhs)
        if not self.best_lhs:
            raise RuntimeError("no converged lhs samples found.")
        print(f"lhs phase complete. best initial design: {self.best_lhs.get('name')} (J={self.best_lhs.get('J'):.4f})")

    def _step_3_active_loop(self):
        print("\n=== step 3: active surrogate refinement loop ===")
        Xfoil_Analysis.DAT_DIR = config.OPTIMAL_DIR / "dat"
        Xfoil_Analysis.INPUTS_DIR = config.OPTIMAL_DIR / "input"
        Xfoil_Analysis.POLAR_DIR = config.OPTIMAL_DIR / "polar"

        for k in range(config.ACTIVE_MAX_ITERS):
            print(f"\n--- active iteration {k + 1} / {config.ACTIVE_MAX_ITERS} ---")
            best_before_iter = self._get_best_record()
            surrogate = train_surrogate_models(self.lhs_results)
            
            seed_idx = best_before_iter.get("params", {}).get("seed_idx", 0)
            current_best_before = best_before_iter.get('J', float('inf'))
            
            new_design = run_gradient_refinement(
                best_lhs=best_before_iter,
                surrogate=surrogate,
                params=self.params,
                seed_airfoil=self.seed_airfoils[seed_idx]["xy"],
                result_name=f"opt_{k+1:02d}",
                lhs_results=self.lhs_results
            )

            self.lhs_results.append(new_design)
            with config.LHS_RESULTS_PATH.open("wb") as f: pickle.dump(self.lhs_results, f)

            # check for convergence
            converged, gap, regret = False, None, None
            if new_design.get("xfoil", {}).get("converged"):
                converged, gap, regret = self._check_advanced_convergence(new_design, surrogate, current_best_before)
            
            # store diagnostics
            self.convergence_diagnostics.append({
                'iteration': k + 1,
                'optimality_gap': gap,
                'max_expected_improvement': regret
            })

            if converged:
                break
        
        self.final_design = self._get_best_record()
        self._copy_final_outputs()
        self._print_final_results()
    
    def _step_4_visualize(self):
        print("\n=== step 4: generating visualizations ===")
        create_all_plots(
            seed_airfoils=self.seed_airfoils,
            best_lhs=self.initial_best_lhs,
            final_design=self.final_design,
            all_lhs_results=self.lhs_results,
            convergence_diagnostics=self.convergence_diagnostics
        )

    def _get_best_record(self):
        converged_results = [r for r in self.lhs_results if r.get("J") is not None]
        return min(converged_results, key=lambda r: r["J"]) if converged_results else {}
    
    def _check_advanced_convergence(self, new_design, surrogate, previous_best_J):
        max_ei = new_design.get('max_ei', float('inf'))
        print(f"max expected improvement (regret) from this iteration: {max_ei:.4e}")
        
        # sample-complexity-based check (optimality gap)
        weights = Weights()
        lcb_values = []
        z_score = norm.ppf(config.CONFIDENCE_LEVEL)

        for record in self.lhs_results:
            coords = record.get('xy')
            if coords is None: continue
            pred_data = surrogate.predict(coords, return_std=True)
            j_mu = score_design(pred_data, coords, weights)
            cl_std = pred_data.get("CL_max_std", 0)
            cd_std = pred_data.get("CD_std", 0)
            alpha_std = pred_data.get("alpha_CL_max_std", 0)
            ld_std = pred_data.get("LD_max_std", 0)
            j_sigma = (weights.w_cl * cl_std + weights.w_cd * cd_std +
                       weights.w_alpha * alpha_std + weights.w_cl * ld_std)
            lcb_values.append(j_mu - z_score * j_sigma)

        optimality_gap = None
        if lcb_values and previous_best_J is not None:
            optimistic_best_J = min(lcb_values)
            current_best_J = previous_best_J
            optimality_gap = current_best_J - optimistic_best_J
            print(f"optimistic best J (LCB): {optimistic_best_J:.4f} | optimality gap: {optimality_gap:.4f}")

        # make convergence decision
        converged = False
        if max_ei < config.EI_REGRET_TOLERANCE:
            print(f"converged: maximum potential regret is below tolerance ({config.EI_REGRET_TOLERANCE}).")
            converged = True
        elif optimality_gap is not None and optimality_gap < config.ACTIVE_J_TOL:
            print(f"converged: optimality gap is below tolerance ({config.ACTIVE_J_TOL}).")
            converged = True
            
        return converged, optimality_gap, max_ei
        
    def _copy_final_outputs(self):
        name = self.final_design.get("name")
        if not name: return
        dat_src = Xfoil_Analysis.DAT_DIR / f"{name}.dat"
        polar_src = Xfoil_Analysis.POLAR_DIR / f"{name}_polar.txt"
        if dat_src.exists(): shutil.copyfile(dat_src, config.OPTIMAL_DIR / "opt_final.dat")
        if polar_src.exists(): shutil.copyfile(polar_src, config.OPTIMAL_DIR / "opt_final_polar.txt")

    def _print_final_results(self):
        xf = self.final_design.get("xfoil", {})
        if not xf or not xf.get("converged"):
            print("\noptimization did not converge to a valid design.")
            return
        
        print("\n" + "=" * 25 + " optimization complete " + "=" * 25)
        print(f"final design: {self.final_design.get('name')} | J={self.final_design.get('J'):.4f}")
        print(f"  cl_max = {xf.get('CL_max'):.4f} at alpha = {xf.get('alpha_CL_max'):.2f} deg")
        print(f"  CD at cl_max = {xf.get('CD'):.5f} | (l/d)_max = {xf.get('LD_max'):.2f}")
        print(f"  cm at cl_max = {xf.get('CM'):.4f} | t/c = {thickness_ratio(self.final_design.get('xy')):.4f}")
