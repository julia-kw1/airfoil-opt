"""
XFOIL analysis runner and polar parser.
"""
import numpy as np
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from . import config

class Analysis_Params:
    """Container for XFOIL analysis settings."""
    def __init__(self):
        self.ALPHA_START = config.ALPHA_START
        self.ALPHA_END = config.ALPHA_END
        self.ALPHA_STEP = config.ALPHA_STEP
        self.MACH = config.MACH_NUMBER
        self.RE_VAL = config.REYNOLDS_NUMBER
        self.PANEL_POINTS = config.PANEL_POINTS

class Xfoil_Analysis:
    """Manages the execution of XFOIL for a single airfoil geometry."""
    DAT_DIR = config.DAT_DIR
    POLAR_DIR = config.POLAR_DIR
    INPUTS_DIR = config.INPUTS_DIR

    def __init__(self, name: str, info: Dict[str, Any], params: Analysis_Params):
        self.name = name
        self.params = params
        self.xy = np.asarray(info.get("xy"), float)
        self.dat_file = self.DAT_DIR / f"{self.name}.dat"
        self.input_file = self.INPUTS_DIR / f"{self.name}_input.txt"
        self.polar_file = self.POLAR_DIR / f"{self.name}_polar.txt"

    @classmethod
    def init_dirs(cls):
        """Reinitialize working directories cleanly."""
        for d in [cls.DAT_DIR, cls.POLAR_DIR, cls.INPUTS_DIR]:
            if d.exists(): shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    def _write_airfoil_dat(self) -> bool:
        if self.xy.ndim != 2 or self.xy.shape[1] != 2:
            return False
        with open(self.dat_file, "w") as f:
            f.write(f"{self.name}\n")
            np.savetxt(f, self.xy, fmt="%.7f")
        return True

    def _write_xfoil_input(self, iter_count: int = 250):
        lines = [
            "PLOP", "G F", "",
            f"LOAD {self.dat_file}",
            "PANE", "PPAR", f"N {self.params.PANEL_POINTS}", "", "",
            "OPER",
            f"Visc {int(self.params.RE_VAL)}",
            f"M {self.params.MACH}",
            f"ITER {iter_count}",
            "PACC", f"{self.polar_file}", "",
            f"ASeq {self.params.ALPHA_START} {self.params.ALPHA_END} {self.params.ALPHA_STEP}",
            "PACC", "", "quit"
        ]
        self.input_file.write_text("\n".join(lines) + "\n")

    def _write_cp_input(self, alpha: float, cp_path: Path, iter_count: int = 200):
        lines = [
            "PLOP", "G F", "",
            f"LOAD {self.dat_file}",
            "PANE", "PPAR", f"N {self.params.PANEL_POINTS}", "", "",
            "OPER",
            f"Visc {int(self.params.RE_VAL)}",
            f"M {self.params.MACH}",
            f"ITER {iter_count}",
            f"ALFA {alpha}",
            "CPWR", f"{cp_path}",
            "", "quit"
        ]
        self.input_file.write_text("\n".join(lines) + "\n")
    
    def _run_xfoil(self, timeout: Optional[int] = 15) -> bool:
        exe = config.XFOIL_EXECUTABLE
        if not shutil.which(exe):
            return False

        with open(self.input_file, "rb") as stdin_f:
            proc = subprocess.Popen([exe], stdin=stdin_f, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                proc.wait(timeout=timeout)
                return proc.returncode == 0
            except subprocess.TimeoutExpired:
                proc.kill()
                return False

    def _parse_polar(self) -> Dict[str, Any]:
        """
        Parses the polar file and returns a dictionary of aerodynamic coefficients.
        """
        empty_result = {
            "CL_max": np.nan, "alpha_CL_max": np.nan, "CD": np.nan, "CM": np.nan,
            "LD_max": np.nan, "Top_Xtr": np.nan, "Bot_Xtr": np.nan,
            "CD_alpha0": np.nan, "CM_alpha0": np.nan, "converged": False
        }
        if not self.polar_file.exists():
            print(f"XFOIL did not produce a polar file for {self.name}; treating as unconverged.")
            return empty_result

        if self.polar_file.stat().st_size == 0:
            print(f"XFOIL polar file empty for {self.name}; treating as unconverged.")
            return empty_result

        try:
            data = np.loadtxt(self.polar_file, skiprows=12)
            if data.size == 0: return empty_result
            
            alpha, cl, cd, cm = data[:, 0], data[:, 1], data[:, 2], data[:, 4]
            ld = cl / np.maximum(cd, 1e-6)
            
            i_CL_max = np.argmax(cl)
            i_ld_max = np.argmax(ld)
            i_alpha0 = np.argmin(np.abs(alpha))
            
            return {
                "CL_max": float(cl[i_CL_max]),
                "alpha_CL_max": float(alpha[i_CL_max]),
                "CD": float(cd[i_CL_max]),
                "CM": float(cm[i_CL_max]),
                "LD_max": float(ld[i_ld_max]),
                "Top_Xtr": float(data[i_CL_max, 5]) if data.shape[1] >= 6 else np.nan,
                "Bot_Xtr": float(data[i_CL_max, 6]) if data.shape[1] >= 7 else np.nan,
                "CD_alpha0": float(cd[i_alpha0]),
                "CM_alpha0": float(cm[i_alpha0]),
                "converged": True
            }
        except (IOError, IndexError, ValueError):
            return empty_result
