"""
Gaussian Process (GP) surrogate model for aerodynamic prediction and its training utilities.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from typing import Dict, Any, List
import pickle
from pathlib import Path
from . import config

# ==============================================================================
# Main Training Function
# ==============================================================================

def train_surrogate_models(lhs_results: List[Dict[str, Any]]) -> 'MultiOutputGP':
    """
    Train a multi-output GP surrogate model from LHS samples.
    This is the main entry point for creating and training the surrogate.
    """
    geometries = [r['xy'] for r in lhs_results]
    aero_data = [r['xfoil'] for r in lhs_results]
    
    print("\n--- Training Gaussian Process Surrogate Models ---")
    surrogate = MultiOutputGP()
    surrogate.fit(geometries, aero_data)
    surrogate.save(config.SURROGATE_MODELS_DIR)
    
    print(f"Surrogate models saved to {config.SURROGATE_MODELS_DIR}/")
    return surrogate

# ==============================================================================
# Gaussian Process Model Classes
# ==============================================================================

class GaussianProcessSurrogate:
    """
    A single-output Gaussian Process Regression model.
    """
    
    def __init__(self, length_scale: float = 1.0, signal_variance: float = 5.0, 
                 noise_variance: float = 1e-4):
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.X_train, self.y_train, self.K_inv = None, None, None
        self.x_scaler_mean, self.x_scaler_std = None, None
        self.y_scaler_mean, self.y_scaler_std = None, None
        self.trained = False
        
    def _squared_exponential_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        dists = cdist(X1, X2, metric='sqeuclidean')
        return self.signal_variance * np.exp(-0.5 * dists / (self.length_scale ** 2))
    
    def _negative_log_marginal_likelihood(self, theta: np.ndarray) -> float:
        self.length_scale, self.signal_variance, self.noise_variance = np.exp(theta)
        K = self._squared_exponential_kernel(self.X_train_scaled, self.X_train_scaled)
        K += self.noise_variance * np.eye(len(self.X_train_scaled))
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train_scaled))
            return 0.5 * self.y_train_scaled.T @ alpha + np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            return 1e10

    def _extract_features(self, coords: np.ndarray) -> np.ndarray:
        i_le = int(np.argmin(coords[:, 0]))
        upper, lower = coords[:i_le + 1][::-1], coords[i_le:]
        x_sample = np.linspace(0, 1, 30)
        y_upper = np.interp(x_sample, upper[:, 0], upper[:, 1])
        y_lower = np.interp(x_sample, lower[:, 0], lower[:, 1])
        return np.concatenate([y_upper, y_lower, [np.max(y_upper - y_lower)]])
    
    def fit(self, geometries: List[np.ndarray], aero_data: List[Dict], target_key: str):
        X, y = [], []
        for coords, result in zip(geometries, aero_data):
            if not (np.all(np.isfinite(coords)) and coords.size > 0):
                continue

            features = self._extract_features(coords)
            target_val = result.get(target_key)
            
            # If the value is valid and finite, use it.
            if target_val is not None and np.isfinite(target_val):
                X.append(features)
                y.append(target_val)
            elif not result.get('converged'):
                X.append(features)
                if 'CD' in target_key:
                    y.append(0.5) # High drag
                else:
                    y.append(0.0) # Low lift, moment, etc.
        # ---
        
        if len(X) < 5:
            raise ValueError(f"Insufficient valid samples for training '{target_key}': {len(X)}")
        
        self.X_train, self.y_train = np.array(X), np.array(y)
        
        self.x_scaler_mean, self.x_scaler_std = self.X_train.mean(axis=0), self.X_train.std(axis=0)
        self.x_scaler_std[self.x_scaler_std < 1e-8] = 1.0
        self.X_train_scaled = (self.X_train - self.x_scaler_mean) / self.x_scaler_std
        
        self.y_scaler_mean, self.y_scaler_std = self.y_train.mean(), self.y_train.std()
        if self.y_scaler_std < 1e-8: self.y_scaler_std = 1.0
        self.y_train_scaled = (self.y_train - self.y_scaler_mean) / self.y_scaler_std
        
        theta0 = np.log([1.0, 5.0, 1e-4])
        bounds = [(-2, 4), (-4, 4), (-12, -0)]
        
        result = minimize(self._negative_log_marginal_likelihood, theta0, method='L-BFGS-B', bounds=bounds)
        self.length_scale, self.signal_variance, self.noise_variance = np.exp(result.x)
        
        K = self._squared_exponential_kernel(self.X_train_scaled, self.X_train_scaled)
        K += self.noise_variance * np.eye(len(self.X_train_scaled))
        self.K_inv = np.linalg.inv(K)
        self.trained = True
        print(f"  GP for {target_key}: l={self.length_scale:.3f}, sig_var={self.signal_variance:.3f}, noise_var={self.noise_variance:.2e}")
    
    def predict(self, coords: np.ndarray, return_std: bool = False):
        if not self.trained: raise RuntimeError("Model not trained.")
        
        x_test = self._extract_features(coords).reshape(1, -1)
        x_test_scaled = (x_test - self.x_scaler_mean) / self.x_scaler_std
        
        k_star = self._squared_exponential_kernel(x_test_scaled, self.X_train_scaled)
        mu_normalized = k_star @ self.K_inv @ self.y_train_scaled
        mu = mu_normalized[0] * self.y_scaler_std + self.y_scaler_mean
        
        if return_std:
            k_star_star = self._squared_exponential_kernel(x_test_scaled, x_test_scaled)
            var = k_star_star - k_star @ self.K_inv @ k_star.T
            std = np.sqrt(np.maximum(var[0, 0], 0)) * self.y_scaler_std
            return float(mu), float(std)
        return float(mu)

class MultiOutputGP:
    """A wrapper that manages multiple single-output GP models."""
    
    def __init__(self):
        self.models: Dict[str, GaussianProcessSurrogate] = {}
        self.targets = ['CL_max', 'alpha_CL_max', 'CD', 'CM', 'LD_max', 'Top_Xtr', 'CD_alpha0']
    
    def fit(self, geometries: List[np.ndarray], aero_data: List[Dict]):
        for target in self.targets:
            gp = GaussianProcessSurrogate()
            try:
                gp.fit(geometries, aero_data, target_key=target)
                self.models[target] = gp
            except (ValueError, np.linalg.LinAlgError) as e:
                print(f"  [Warning] Could not train model for '{target}': {e}")
    
    def predict(self, coords: np.ndarray, return_std: bool = False) -> Dict[str, Any]:
        results = {}
        for target, gp in self.models.items():
            try:
                if return_std:
                    pred, std = gp.predict(coords, return_std=True)
                    results[target], results[f"{target}_std"] = pred, std
                else:
                    results[target] = gp.predict(coords, return_std=False)
            except RuntimeError:
                results[target] = np.nan
                if return_std: results[f"{target}_std"] = np.nan
        return results
    
    def save(self, directory: Path):
        directory.mkdir(exist_ok=True)
        for target, gp in self.models.items():
            with open(directory / f"{target}_gp.pkl", 'wb') as f:
                pickle.dump(gp, f)
    
    @staticmethod
    def load(directory: Path) -> 'MultiOutputGP':
        multi_gp = MultiOutputGP()
        for target in multi_gp.targets:
            filepath = directory / f"{target}_gp.pkl"
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    multi_gp.models[target] = pickle.load(f)
        return multi_gp
