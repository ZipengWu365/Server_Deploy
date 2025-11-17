# -*- coding: utf-8 -*-
"""
ECHO STATE NETWORK (ESN) - IN-MEMORY PRE-EXTRACTION VERSION
Target Environment: High-memory (96GB+) Linux server.

This version replaces the CNN model with an Echo State Network (Reservoir Computing).
Training is not iterative (no SGD) but done by solving a single Ridge Regression
problem on the collected reservoir states.
"""

import os, gc
import psutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ESN 依赖
import scipy.linalg
from scipy.sparse import rand as sparse_rand
from scipy.linalg import pinv, solve

import time, json

# ============================ Server-Optimized Config ============================
# 🔑 Key parameters for server environment
DATA_DIR = "./dataset"
DATASETS = [
    "exchange_rate.csv","ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv",
    # "electricity.csv", "weather.csv","traffic.csv",
]
HORIZONS = [96, 192, 336, 720]

LOOKBACK_WINDOW = 336

# Algorithm and model selection
ALGO = 'ESN'  # Echo State Network
MODEL_TYPE = 'reservoir'

# --- ESN Hyperparameters ---
ESN_RESERVOIR_SIZE = 500     # 储备池中的神经元数量
ESN_SPECTRAL_RADIUS = 0.99   # 储备池权重矩阵的光谱半径
ESN_SPARSITY = 0.1           # 储备池的稀疏度 (10% a
ESN_RIDGE_ALPHA = 1.0        # 读出层岭回归的L2惩罚项
ESN_INPUT_SCALING = 1.0      # 输入权重的缩放
# -----------------------------

# Server-optimized settings
N_JOBS = -1 # Use all available CPU cores (for Scipy if applicable)

STANDARD_SPLITS = {
    'etth1': (8545, 2881, 2881),
    'etth2': (8545, 2881, 2881),
    'ettm1': (34465, 11521, 11521),
    'ettm2': (34465, 11521, 11521),
    'electricity': (18317, 2633, 5261),
    'traffic': (12185, 1757, 3509),
    'weather': (36792, 5271, 10540),
    'exchange_rate': (5120, 665, 1422),
}

# ======================= Memory Monitoring Utils =======================
def get_memory_usage_gb():
    """Gets current process memory usage in GB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**3
    except:
        return 0.0

def print_memory(label: str = ""):
    """Prints memory usage."""
    mem_gb = get_memory_usage_gb()
    if mem_gb > 0:
        print(f"    💾 [Memory] {label}: {mem_gb:.2f} GB")

# ======================= IO & utils =======================
def load_csv(path: str) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path, header=0, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce').ffill().fillna(0.0)
    return df.values.T.astype(np.float32), list(df.columns)

def zscore_channelwise(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=1, keepdims=True, dtype=np.float32)
    std = X.std(axis=1, keepdims=True, dtype=np.float32) + 1e-8
    return (X - mu) / std, mu.squeeze(), std.squeeze()

def split_by_standard(X: np.ndarray, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    name = os.path.basename(dataset_name).lower()
    key = next((k for k in STANDARD_SPLITS if k in name), None)
    ch, T = X.shape
    if key and sum(STANDARD_SPLITS[key]) <= T:
        tr, va, te = STANDARD_SPLITS[key]
        return X[:, :tr], X[:, tr:tr+va], X[:, tr+va:tr+va+te], True
    else:
        t1, t2 = int(T * 0.7), int(T * 0.85)
        return X[:, :t1], X[:, t1:t2], X[:, t2:], False

# ======================= Echo State Network (ESN) Forecaster =======================

class ESNForecaster:
    """
    ESNForecaster
    Replaces the PyTorch loop with ESN's state collection and
    Ridge Regression readout.
    
    This implementation treats the `lookback` window as a sequence
    to generate a single feature vector (the final reservoir state).
    """
    def __init__(self,
                 lookback: int,
                 n_reservoir: int,
                 spectral_radius: float,
                 sparsity: float,
                 ridge_alpha: float,
                 input_scaling: float,
                 random_state: int = 42):
        
        self.lookback = int(lookback)
        self.n_reservoir = int(n_reservoir)
        self.spectral_radius = float(spectral_radius)
        self.sparsity = float(sparsity)
        self.ridge_alpha = float(ridge_alpha)
        self.input_scaling = float(input_scaling)
        self.rng = np.random.default_rng(random_state)
        
        self.W_in: Optional[np.ndarray] = None
        self.W_res: Optional[np.ndarray] = None
        self.W_out: Dict[int, np.ndarray] = {}
        self.scalers: Dict[int, StandardScaler] = {}

    def _init_weights(self):
        """Initializes the fixed ESN weights (W_in and W_res)."""
        print("  [ESN] Initializing reservoir weights...")
        
        # 1. Input weights W_in: (reservoir_size, 1) since we feed 1 step at a time
        self.W_in = self.rng.uniform(-1, 1, (self.n_reservoir, 1)) * self.input_scaling
        
        # 2. Reservoir weights W_res: (reservoir_size, reservoir_size)
        # Create a sparse random matrix
        W = sparse_rand(self.n_reservoir, self.n_reservoir,
                        density=self.sparsity,
                        format='csr',
                        random_state=self.rng)
        # Convert to dense and adjust values from [0, 1] to [-0.5, 0.5]
        self.W_res = W.toarray() - 0.5
        
        # Rescale spectral radius (largest absolute eigenvalue)
        try:
            eigvals = np.linalg.eigvals(self.W_res)
            current_radius = np.max(np.abs(eigvals))
            if current_radius > 1e-9:
                 self.W_res *= (self.spectral_radius / current_radius)
            print(f"  [ESN] Reservoir initialized. Spectral radius: {np.max(np.abs(np.linalg.eigvals(self.W_res))):.4f}")
        except np.linalg.LinAlgError:
            print("  [ESN] WARNING: Eigenvalue computation failed. Using unscaled reservoir.")

    def _get_states(self, X_data_scaled: np.ndarray) -> np.ndarray:
        """
        Feeds the input data (scaled) through the reservoir to collect
        the final states.
        
        Args:
            X_data_scaled: np.ndarray of shape (n_samples, lookback)
            
        Returns:
            H_all: np.ndarray of shape (n_samples, n_reservoir)
        """
        n_samples = X_data_scaled.shape[0]
        H_all = np.zeros((n_samples, self.n_reservoir), dtype=np.float32)
        
        for i in range(n_samples):
            x_seq = X_data_scaled[i, :] # Shape (lookback,)
            h = np.zeros(self.n_reservoir, dtype=np.float32) # Reset state
            
            # Unroll the lookback window sequentially through the reservoir
            for t in range(self.lookback):
                u = x_seq[t:t+1] # Shape (1,)
                # ESN state update equation
                h = np.tanh(self.W_in @ u + self.W_res @ h)
            
            H_all[i, :] = h # Store the final state
        
        return H_all

    def fit(self, ridge_train_data: np.ndarray,
            val_data: np.ndarray, horizons: List[int]):
        
        # 1. Initialize reservoir weights (once)
        self._init_weights()
        
        n_channels, T_train = ridge_train_data.shape
        
        for horizon in horizons:
            print(f"\n  Training ESN model, horizon {horizon}...")
            print_memory(f"Before training H={horizon}")
            min_len = self.lookback + horizon

            # ========== Phase 1: Pre-extract all training samples ==========
            print(f"    Phase 1: Pre-extracting all training samples to RAM...")
            X_train_list, Y_train_list = [], []
            
            for ch in range(n_channels):
                series = ridge_train_data[ch]
                n = max(0, T_train - min_len + 1)
                if n == 0: continue
                for i in range(n):
                    win = series[i:i+self.lookback]
                    target = series[i+self.lookback:i+self.lookback+horizon]
                    X_train_list.append(win)
                    Y_train_list.append(target)
            
            if not X_train_list:
                print("    [skip] Not enough training data length for this horizon.")
                continue
                
            X_train_raw = np.array(X_train_list, dtype=np.float32)
            Y_train = np.array(Y_train_list, dtype=np.float32)
            del X_train_list, Y_train_list
            gc.collect()
            
            print(f"    Total train samples: {len(X_train_raw)}")
            print_memory("After train sample extraction")

            # ========== Phase 2: Fit Scaler and Scale Data ==========
            print(f"    Phase 2: Fitting scaler and scaling data...")
            scaler = StandardScaler().fit(X_train_raw)
            X_train_scaled = scaler.transform(X_train_raw)
            self.scalers[horizon] = scaler # Save scaler
            
            # ========== Phase 3: Collect Reservoir States ==========
            print(f"    Phase 3: Collecting reservoir states...")
            H_train = self._get_states(X_train_scaled)
            del X_train_raw, X_train_scaled
            gc.collect()
            print_memory("After state collection")
            
            # ========== Phase 4: Train Readout Layer (Ridge Regression) ==========
            print(f"    Phase 4: Training readout layer...")
            try:
                # Solve (H_T @ H + alpha*I) @ W_out = H_T @ Y
                # This is more stable than calculating the pseudo-inverse directly
                H_T = H_train.T
                I = np.identity(self.n_reservoir, dtype=np.float32)
                
                # Solve the linear system
                W_out = solve(
                    H_T @ H_train + self.ridge_alpha * I,
                    H_T @ Y_train,
                    assume_a='pos' # Assume a positive definite matrix
                )
                self.W_out[horizon] = W_out.astype(np.float32)
                
            except np.linalg.LinAlgError:
                print(f"    [Error] Linear algebra solve failed for H={horizon}. Falling back to pseudo-inverse.")
                H_T = H_train.T
                I = np.identity(self.n_reservoir, dtype=np.float32)
                self.W_out[horizon] = (pinv(H_T @ H_train + self.ridge_alpha * I) @ H_T @ Y_train).astype(np.float32)
                
            print(f"    ✓ Training complete for H={horizon}")
            del H_train, Y_train
            gc.collect()

    def predict(self, test_data: np.ndarray, horizon: int) -> List[np.ndarray]:
        if horizon not in self.W_out:
            raise ValueError(f"Model not trained for horizon {horizon}")

        W_out = self.W_out[horizon]
        scaler = self.scalers[horizon]
        
        n_channels, T = test_data.shape
        preds = []
        min_len = self.lookback + horizon

        for ch in range(n_channels):
            series = test_data[ch]
            nsamp = max(0, T - min_len + 1)
            if nsamp == 0:
                preds.append(np.array([], dtype=np.float32))
                continue

            # 1. Extract raw test samples
            X_list = [series[i:i+self.lookback] for i in range(nsamp)]
            X_test_raw = np.array(X_list, dtype=np.float32)
            
            # 2. Scale test samples
            X_test_scaled = scaler.transform(X_test_raw)
            
            # 3. Get reservoir states for test data
            H_test = self._get_states(X_test_scaled)
            
            # 4. Predict using W_out
            Y_pred = H_test @ W_out
            
            preds.append(Y_pred.astype(np.float32))
            del X_list, X_test_raw, X_test_scaled, H_test

        return preds

# ======================= Evaluation =======================
def evaluate_train_only(file_path: str, lookback_default: int = LOOKBACK_WINDOW,
                        horizons: List[int] = HORIZONS) -> Dict:
    dataset_name = os.path.basename(file_path)
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")

    # ESN doesn't use the per-dataset config, just global lookback
    lookback = lookback_default
    print(f"Config: lookback={lookback}")

    Xraw, _ = load_csv(file_path)
    Xs, _, _ = zscore_channelwise(Xraw) # ❗ 提醒：这里仍有数据泄露问题
    ch, T = Xs.shape
    X_train, X_val, X_test, used_standard = split_by_standard(Xs, file_path)
    print(f"Data: {ch} channels × {T} timesteps")
    print(f"Split: {'standard' if used_standard else 'ratio'}")
    print(f"Protocol: Model on TRAIN")

    # 实例化 ESNForecaster
    model = ESNForecaster(
        lookback=lookback,
        n_reservoir=ESN_RESERVOIR_SIZE,
        spectral_radius=ESN_SPECTRAL_RADIUS,
        sparsity=ESN_SPARSITY,
        ridge_alpha=ESN_RIDGE_ALPHA,
        input_scaling=ESN_INPUT_SCALING
    )

    # val_data is not used for ESN training, but passed for API compatibility
    model.fit(ridge_train_data=X_train,
             val_data=X_val, horizons=horizons)

    results = {}
    for horizon in horizons:
        print(f"\n  Predicting horizon {horizon}...")
        predictions = model.predict(X_test, horizon)
        
        total_sse = 0.0
        total_sae = 0.0
        total_n = 0
        for ch_idx in range(ch):
            series = X_test[ch_idx]
            nsamp = max(0, series.shape[0] - (lookback + horizon) + 1)
            if nsamp <= 0:
                continue
            Y_true = np.empty((nsamp, horizon), dtype=np.float32)
            for i in range(nsamp):
                s = i + lookback
                Y_true[i, :] = series[s: s + horizon]
            Y_hat = predictions[ch_idx]
            if Y_hat.size == 0:
                continue
            diff = (Y_hat - Y_true).astype(np.float32, copy=False)
            total_sse += float(np.sum(diff * diff))
            total_sae += float(np.sum(np.abs(diff)))
            total_n += diff.size
        mse = total_sse / total_n if total_n > 0 else np.nan
        mae = total_sae / total_n if total_n > 0 else np.nan
        results[horizon] = {'mse': mse, 'mae': mae}
        print(f"  [{ALGO}-{MODEL_TYPE.upper()}] H={horizon:3d} | MSE={mse:.4f}  MAE={mae:.4f}")

    gc.collect()
    return results

# ======================= Main =======================
def run_evaluation():
    print("="*80)
    print(f"{ALGO}-{MODEL_TYPE.upper()} Model - Server Optimized (In-Memory)")
    print("="*80)
    print(f"Reservoir Size: {ESN_RESERVOIR_SIZE} | Spectral Radius: {ESN_SPECTRAL_RADIUS} | Sparsity: {ESN_SPARSITY}")
    print(f"Model: TRAIN (no validation set needed for ESN training)")
    print("="*80)

    all_results = []

    for dataset in DATASETS:
        filepath = os.path.join(DATA_DIR, dataset)
        if not os.path.exists(filepath):
            print(f"[Skip] {filepath} not found")
            continue
        try:
            results = evaluate_train_only(filepath)

            row = {
                'dataset': dataset,
                'model': f'{ALGO}_{MODEL_TYPE}',
            }

            for horizon in HORIZONS:
                row[f'mse_{horizon}'] = results[horizon]['mse']
                row[f'mae_{horizon}'] = results[horizon]['mae']

            all_results.append(row)

        except Exception as e:
            print(f"[Error] Failed on {dataset}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        df_results = pd.DataFrame(all_results)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"{ALGO}_{MODEL_TYPE}_{timestamp}.csv"
        df_results.to_csv(output_file, index=False)

        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(df_results.to_string())
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    print("System Configuration:")
    print(f"  Numpy: {np.__version__}")
    print(f"  Scipy: {scipy.__version__}")
    print()

    run_evaluation()