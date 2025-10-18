# -*- coding: utf-8 -*-
"""
SERVER-OPTIMIZED SSA/STD CNN - IN-MEMORY PRE-EXTRACTION VERSION
Target Environment: High-memory (96GB+) Linux server with CUDA GPU.

Optimizations:
- C++ accelerated SSA/STD via fasttimes.
- **In-Memory Feature Pre-extraction**: Extracts all features into RAM once before training.
  This is the fastest method, suitable for high-memory machines, as it eliminates
  both re-computation during epochs and disk I/O from chunking.
- Aggressive memory cleanup.
- CUDA-optimized tensor operations.
"""

import os, gc, re
import psutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from types import MethodType
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# C++ accelerated SSA/STD functions
from fasttimes import fastssa, faststd
import time, json

# ============================ Server-Optimized Config ============================
# 🔑 Key parameters for server environment
CHUNK_SIZE_CHANNELS = 100  # Increased for faster basis learning
MAX_SAMPLES_SSA = 100000 # Increased for more stable SSA basis
MAX_ROWS_STD = 200000 # Increased for more stable STD basis

DATA_DIR = "./dataset"
DATASETS = [
    "exchange_rate.csv","traffic.csv","ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv",
    "electricity.csv", "weather.csv",
]
HORIZONS = [96, 192, 336, 720]

LOOKBACK_WINDOW = 336

# Algorithm and model selection
ALGO = 'SSA'  # 'STD' or 'SSA'
LEARN_BASES_ON = 'train'  # 'train+val' or 'train'
MODEL_TYPE = 'cnn'  # 'linear' or 'cnn'

# Server-optimized settings
BUILD_BATCH = 8000
N_JOBS = -1 # Use all available CPU cores

# Device selection: Prioritize CUDA
def get_device():
    if torch.cuda.is_available():
        print("✓ Using CUDA GPU")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("✓ Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    else:
        print("⚠ Using CPU")
        return torch.device("cpu")

DEVICE = get_device()

# Neural network hyperparameters - optimized for high-memory GPU
BATCH_SIZE = 1024      # Increased from 256
EPOCHS = 100
LEARNING_RATE = 0.005  # Slightly reduced for larger batches
L2_WEIGHT_DECAY = 1.0
PATIENCE = 15
LR_SCHEDULER = True

# Multi-scale params
DATASET_PARAMS = {
    'etth': {'block_sizes': [24, 168, 504, 1440], 'seasonal_ranks': [6, 5]},
    'ettm': {'block_sizes': [96, 672, 1440], 'seasonal_ranks': [8, 6]},
    'electricity': {'block_sizes': [24, 168, 1440], 'seasonal_ranks': [8, 6]},
    'traffic': {'block_sizes': [24, 168, 1440], 'seasonal_ranks': [10, 4]},
    'weather': {'block_sizes': [24, 168, 1440], 'seasonal_ranks': [6, 4]},
    'exchange_rate': {'block_sizes': [7, 30, 1440], 'seasonal_ranks': [2, 3]},
}

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

DEFAULT_GROUP_MULTIPLIERS_BASE = {
    'raw': 1.25,
    'trend_dyn': 0.7,
    'stats': 1.0,
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

# ======================= Server-Optimized PyTorch Models =======================
class LinearL2(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class CNN1D(nn.Module):
    """1D CNN with slightly increased capacity for server."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # Increased channels: 16->32, 32->64, 16->32
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        conv_out_len = input_dim // 2
        self.fc1 = nn.Linear(32 * conv_out_len, 128) # Increased from 64
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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

# ======================= STD core =======================
@dataclass
class STDComponents:
    trend: np.ndarray
    dispersion: np.ndarray
    seasonal: np.ndarray
    block_size: int
    n_blocks: int

class STDDecomposer:
    def __init__(self, block_size: int, seasonal_rank: int, ev_thresh: float = 0.92,
                 r_min: int = 1, r_max_cap: int = 12):
        self.block_size = int(block_size)
        self.seasonal_rank = int(seasonal_rank)
        self.ev_thresh = float(ev_thresh)
        self.r_min = int(r_min)
        self.r_max_cap = int(r_max_cap)
        self.seasonal_basis = None

    def decompose(self, X: np.ndarray) -> Optional[STDComponents]:
        """C++ accelerated STD decomposition"""
        n_channels, T = X.shape
        bs = self.block_size
        n_blocks = T // bs
        if n_blocks == 0:
            return None

        X_forC = X.astype(np.float32, copy=False)
        trend_c, disp_c, _ = faststd(X_forC, block_size=bs, seasonal_rank=self.seasonal_rank)

        trend = trend_c.T
        dispersion = disp_c.T

        X_trim = X[:, :n_blocks * bs]
        Xb = X_trim.reshape(n_channels, n_blocks, bs).transpose(1, 0, 2).astype(np.float32, copy=False)
        Xc = Xb - trend[:, :, None]
        seasonal = np.zeros_like(Xc, dtype=np.float32)
        mask = dispersion > 1e-6
        seasonal[mask] = Xc[mask] / dispersion[mask][:, None]

        return STDComponents(trend.astype(np.float32), dispersion.astype(np.float32), seasonal, bs, n_blocks)

    def fit_seasonal_basis(self, components: STDComponents):
        S_flat = components.seasonal.reshape(-1, self.block_size).astype(np.float32, copy=False)
        if S_flat.shape[0] > MAX_ROWS_STD:
            idx = np.random.choice(S_flat.shape[0], MAX_ROWS_STD, replace=False)
            S_flat = S_flat[idx]
        U, s, Vt = np.linalg.svd(S_flat, full_matrices=False)
        if self.seasonal_rank < 0:
            ev = (s ** 2)
            total = ev.sum() + 1e-12
            cum = np.cumsum(ev) / total
            r = int(np.searchsorted(cum, self.ev_thresh) + 1)
            r = max(self.r_min, min(r, min(self.block_size, self.r_max_cap)))
            self.seasonal_rank = r
        else:
            r = min(self.seasonal_rank, Vt.shape[0])
            self.seasonal_rank = r
        self.seasonal_basis = Vt[:r, :].T.astype(np.float32, copy=False)
        return self.seasonal_basis

# ======================= SSA core (C++ ACCELERATED) =======================
class SSAProjector:
    def __init__(self, L: int, seasonal_rank: int, r_trend: int = 1, max_samples: int = MAX_SAMPLES_SSA, seed: int = 0):
        self.L = int(L)
        self.r_trend = max(1, int(r_trend))
        self.r_seasonal = max(0, int(seasonal_rank))
        self.max_samples = int(max_samples)
        self.r_total = self.r_trend + self.r_seasonal
        self.U = None
        self.r_actual = 0
        self.rng = np.random.default_rng(seed)
        self._cpp_accelerated = True

    def fit_from_series(self, X: np.ndarray):
        """Learn SSA basis - C++ accelerated reconstruction, Python SVD for basis"""
        n_channels, T = X.shape
        if T < self.L:
            self.U = None; self.r_actual = 0; return

        X_reconstructed = fastssa(X.astype(np.float32), L=self.L, r=self.r_total)

        n_per_ch = T - self.L + 1
        total = n_channels * max(0, n_per_ch)
        if total <= 0:
            self.U = None; self.r_actual = 0; return

        take = min(total, self.max_samples)
        idx = self.rng.choice(total, size=take, replace=False)
        S = np.empty((self.L, take), dtype=np.float32)

        for j, flat in enumerate(idx):
            ch = flat // n_per_ch
            st = flat % n_per_ch
            seg = X_reconstructed[ch, st:st+self.L]
            S[:, j] = seg - float(seg.mean())

        U, s, Vt = np.linalg.svd(S, full_matrices=False)
        r = min(self.r_total, U.shape[1])
        self.U = U[:, :r].astype(np.float32, copy=False)
        self.r_actual = int(r)

        del S, U, s, Vt, X_reconstructed
        gc.collect()

    def project(self, x_window_1d: np.ndarray):
        if self.U is None or x_window_1d.shape[0] < self.L:
            return None, None, None
        seg = x_window_1d[-self.L:].astype(np.float32, copy=False)
        mu = float(seg.mean())
        seg_dm = seg - mu
        energy = float(np.linalg.norm(seg_dm) + 1e-8)
        coef = self.U.T @ seg_dm
        return coef, mu, energy

# ======================= Base Forecaster (Server-Optimized) =======================
class BaseForecaster:
    def __init__(self):
        self.models: Dict[int, nn.Module] = {}
        self.feature_scalers: Dict[int, StandardScaler] = {}
        self.group_weight_vec: Dict[int, np.ndarray] = {}
        self.n_features_: Optional[int] = None
        self.group_slices: Dict[str, slice] = {}
        self._zero_masks: Dict[int, np.ndarray] = {}
        self.slow_means_K: int = 3

    def _infer_feature_dim_and_groups(self, series: np.ndarray, extractor):
        if self.n_features_ is None:
            f = extractor(series[:self.lookback], collect_slices=True)
            self.n_features_ = int(f.shape[0])

    def _build_and_fit_model(self, train_data: np.ndarray, val_data: np.ndarray,
                            horizons: List[int], extractor, weight_maker):
        """
        🚀 IN-MEMORY OPTIMIZATION (for High-Memory Servers):
        1. Pre-extracts ALL features and targets into large NumPy arrays in RAM.
        2. Trains directly from these arrays.
        This is the fastest approach if memory allows (e.g., >96GB).
        """
        n_channels, T_train = train_data.shape
        _, T_val = val_data.shape

        for horizon in horizons:
            print(f"\n  Training {MODEL_TYPE.upper()} model, horizon {horizon}...")
            print_memory(f"Before training H={horizon}")

            min_len = self.lookback + horizon

            # ========== Phase 1: Pre-extract all training features to RAM ==========
            print(f"    Phase 1: Pre-extracting all training features to RAM...")
            X_train_list, Y_train_list = [], []
            total_train_samples = sum(max(0, T_train - min_len + 1) for _ in range(n_channels))

            if total_train_samples == 0:
                print("    [skip] Not enough training data length for this horizon.")
                continue

            # Process in chunks to build the list, to be slightly more memory-friendly during list extension
            for ch_start in range(0, n_channels, CHUNK_SIZE_CHANNELS):
                ch_end = min(ch_start + CHUNK_SIZE_CHANNELS, n_channels)
                for ch in range(ch_start, ch_end):
                    series = train_data[ch]
                    n = max(0, T_train - min_len + 1)
                    if n == 0: continue
                    for i in range(n):
                        win = series[i:i+self.lookback]
                        features = extractor(win)
                        target = series[i+self.lookback:i+self.lookback+horizon]
                        X_train_list.append(features)
                        Y_train_list.append(target)
            
            X_train_all = np.array(X_train_list, dtype=np.float32)
            Y_train_all = np.array(Y_train_list, dtype=np.float32)
            del X_train_list, Y_train_list
            gc.collect()
            
            print(f"    Total train samples: {len(X_train_all)}")
            print_memory("After train feature extraction to RAM")

            # ========== Phase 2: Extract validation set (in-memory) ==========
            print(f"    Phase 2: Extracting validation features...")
            X_val_list, Y_val_list = [], []
            total_val_samples = sum(max(0, T_val - min_len + 1) for _ in range(n_channels))

            if total_val_samples > 0:
                for ch in range(n_channels):
                    series = val_data[ch]
                    n = max(0, T_val - min_len + 1)
                    if n == 0: continue
                    for i in range(n):
                        win = series[i:i+self.lookback]
                        features = extractor(win)
                        target = series[i+self.lookback:i+self.lookback+horizon]
                        X_val_list.append(features)
                        Y_val_list.append(target)
                X_val = np.array(X_val_list, dtype=np.float32)
                Y_val = np.array(Y_val_list, dtype=np.float32)
            else:
                # Fallback: use a part of the training data for validation
                print(f"    ⚠ Val set too short for H={horizon}, using a part of train data for validation.")
                val_split_idx = int(0.9 * len(X_train_all))
                X_val, Y_val = X_train_all[val_split_idx:], Y_train_all[val_split_idx:]
                X_train_all, Y_train_all = X_train_all[:val_split_idx], Y_train_all[:val_split_idx]

            if len(X_val) == 0:
                print("    [skip] Validation set is empty even after fallback.")
                continue
            
            del X_val_list, Y_val_list
            gc.collect()
            print(f"    Val samples: {len(X_val)}")
            print_memory("After validation extraction")

            # ========== Phase 3: Fit Scaler and create DataLoaders ==========
            print("    Phase 3: Fitting scaler and creating DataLoaders...")
            scaler = StandardScaler().fit(X_train_all)
            w, zero_mask = weight_maker(horizon)

            X_train_scaled = scaler.transform(X_train_all)
            X_train_scaled /= w[None, :]
            if zero_mask is not None and zero_mask.any():
                X_train_scaled[:, zero_mask] = 0.0
            
            X_val_scaled = scaler.transform(X_val)
            X_val_scaled /= w[None, :]
            if zero_mask is not None and zero_mask.any():
                X_val_scaled[:, zero_mask] = 0.0

            train_dataset = TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(Y_train_all))
            val_dataset = TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(Y_val))
            
            # Use more workers on server
            num_workers = min(os.cpu_count() // 2, 8)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
            
            del X_train_all, Y_train_all, X_train_scaled, X_val, Y_val, X_val_scaled
            gc.collect()
            print_memory("After DataLoader creation")

            # ========== Phase 4: Model Training ==========
            print("    Phase 4: Starting model training...")
            if MODEL_TYPE == 'linear':
                model = LinearL2(self.n_features_, horizon).to(DEVICE)
            else:
                model = CNN1D(self.n_features_, horizon).to(DEVICE)

            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT_DECAY)
            if LR_SCHEDULER:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

            best_val_loss = float('inf')
            patience_counter = 0
            best_state = None

            for epoch in range(EPOCHS):
                model.train()
                epoch_train_loss = 0.0
                for X_batch, Y_batch in train_loader:
                    X_batch, Y_batch = X_batch.to(DEVICE, non_blocking=True), Y_batch.to(DEVICE, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(X_batch)
                    loss = criterion(outputs, Y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_train_loss += loss.item() * X_batch.size(0)
                epoch_train_loss /= len(train_loader.dataset)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, Y_batch in val_loader:
                        X_batch, Y_batch = X_batch.to(DEVICE, non_blocking=True), Y_batch.to(DEVICE, non_blocking=True)
                        outputs = model(X_batch)
                        loss = criterion(outputs, Y_batch)
                        val_loss += loss.item() * X_batch.size(0)
                val_loss /= len(val_loader.dataset)

                if LR_SCHEDULER: scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"      Early stopping at epoch {epoch+1}")
                        break
                
                if (epoch + 1) % 5 == 0:
                    print(f"      Epoch {epoch+1:3d}/{EPOCHS} | Train: {epoch_train_loss:.4f} | Val: {val_loss:.4f}")
            
            if best_state:
                model.load_state_dict(best_state)
                model.to(DEVICE)

            self.models[horizon] = model
            self.feature_scalers[horizon] = scaler
            self.group_weight_vec[horizon] = w
            self._zero_masks[horizon] = zero_mask if zero_mask is not None else np.zeros_like(w, dtype=bool)

            del train_loader, val_loader, train_dataset, val_dataset
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
            if DEVICE.type == 'mps': torch.mps.empty_cache()
            gc.collect()

            print(f"    ✓ Training complete | Best val loss: {best_val_loss:.4f}")
            print_memory(f"After H={horizon} training")

    def predict(self, test_data: np.ndarray, horizon: int, extractor) -> List[np.ndarray]:
        if horizon not in self.models:
            raise ValueError(f"Model not trained for horizon {horizon}")

        n_channels, T = test_data.shape
        preds = []
        model = self.models[horizon]
        model.eval()
        scaler = self.feature_scalers[horizon]
        w = self.group_weight_vec[horizon]
        zero_mask = self._zero_masks.get(horizon, np.zeros_like(w, dtype=bool))
        min_len = self.lookback + horizon

        for ch in range(n_channels):
            series = test_data[ch]
            nsamp = max(0, T - min_len + 1)
            if nsamp == 0:
                preds.append(np.array([], dtype=np.float32))
                continue

            X_list = [extractor(series[i:i+self.lookback]) for i in range(nsamp)]
            X_test = np.array(X_list, dtype=np.float32)
            X_test_scaled = scaler.transform(X_test)
            X_test_scaled /= w[None, :]

            if zero_mask.any():
                X_test_scaled[:, zero_mask] = 0.0

            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)
                Y_pred = model(X_tensor).cpu().numpy().astype(np.float32)

            preds.append(Y_pred)
            del X_tensor, X_test, X_test_scaled

        return preds

# ======================= STD Forecaster =======================
class STDNeuralForecaster(BaseForecaster):
    def __init__(self, lookback: int, block_sizes: List[int], seasonal_ranks: List[int],
                 group_multipliers: Optional[Dict[str, float]] = None):
        super().__init__()
        if len(block_sizes) != len(seasonal_ranks):
            raise ValueError("block_sizes and seasonal_ranks must have the same length")
        self.lookback = int(lookback)
        self.block_sizes = [int(b) for b in block_sizes]
        self.seasonal_ranks = [int(r) for r in seasonal_ranks]
        self.std_decomposers = {bs: STDDecomposer(bs, sr) for bs, sr in zip(self.block_sizes, self.seasonal_ranks)}
        self._primary_bs = self.block_sizes[0]
        self.group_multipliers_base = self._default_group_multipliers(group_multipliers)

    def _default_group_multipliers(self, user_override: Optional[Dict[str, float]]) -> Dict[str, float]:
        m = dict(DEFAULT_GROUP_MULTIPLIERS_BASE)
        for bs in self.block_sizes:
            m[f'scale_{bs}_trend'] = m.get(f'scale_{bs}_trend', 0.8)
            m[f'scale_{bs}_disp'] = m.get(f'scale_{bs}_disp', 1.0)
            m[f'scale_{bs}_seasonal'] = m.get(f'scale_{bs}_seasonal', 1.0)
        if user_override:
            m.update(user_override)
        for k in list(m.keys()):
            if m[k] <= 0:
                m[k] = 1e-6
        return m

    def _extract_features_single(self, x_window_1d: np.ndarray, collect_slices: bool = False):
        pieces: List[np.ndarray] = []
        names: List[str] = []
        L = x_window_1d.shape[0]

        arr = x_window_1d.astype(np.float32, copy=False).ravel()
        pieces.append(arr); names.append('raw')

        for bs, dec in self.std_decomposers.items():
            if L >= bs:
                xb = x_window_1d[-bs:].reshape(1, -1)
                comp = dec.decompose(xb)
                if comp is not None:
                    pieces.append(comp.trend.astype(np.float32, copy=False).ravel())
                    names.append(f'scale_{bs}_trend')
                    pieces.append(comp.dispersion.astype(np.float32, copy=False).ravel())
                    names.append(f'scale_{bs}_disp')
                    if dec.seasonal_basis is not None:
                        seasonal_enc = comp.seasonal.reshape(1, bs) @ dec.seasonal_basis
                        pieces.append(seasonal_enc.astype(np.float32, copy=False).ravel())
                        names.append(f'scale_{bs}_seasonal')

        pbs = self._primary_bs
        if L >= pbs * 2:
            nb = L // pbs
            x_trim = x_window_1d[:nb*pbs].reshape(nb, pbs).astype(np.float32, copy=False)
            trend_series = x_trim.mean(axis=1)
            t = np.arange(trend_series.shape[0], dtype=np.float32)
            t_mean = t.mean(); y_mean = trend_series.mean()
            denom = np.sum((t - t_mean) ** 2) + 1e-8
            slope = np.sum((t - t_mean) * (trend_series - y_mean)) / denom
            pieces.append(np.array([slope], dtype=np.float32))
            names.append('trend_dyn')

        xw = x_window_1d.astype(np.float32, copy=False)
        q25 = np.percentile(xw, 25).astype(np.float32)
        q75 = np.percentile(xw, 75).astype(np.float32)
        pieces.append(np.array([xw.mean(), xw.std(), xw.min(), xw.max(), q25, q75], dtype=np.float32))
        names.append('stats')

        vec = np.concatenate(pieces, dtype=np.float32)
        if collect_slices:
            self.group_slices.clear()
            pos = 0
            for piece, name in zip(pieces, names):
                n = piece.shape[0]
                self.group_slices[name] = slice(pos, pos + n)
                pos += n
        return vec

    def _make_group_weight_vec_and_zero_mask(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        w = np.ones((self.n_features_,), dtype=np.float32)
        zero_mask = np.zeros((self.n_features_,), dtype=bool)
        eps = 1e-8
        for g, slc in self.group_slices.items():
            m = self.group_multipliers_base.get(g, 1.0)
            if g == 'raw':
                L0 = 168.0
                gamma = 0.5
                m = m * max(1.0, (self.lookback / L0) ** gamma)
            elif g.endswith('_seasonal'):
                eta = 0.4
                try:
                    bs = int(g.split('_')[1])
                    if horizon >= 2*bs:
                        zero_mask[slc] = True
                        m = m * 1e3
                    else:
                        m = m * (1.05 ** ((horizon / max(bs, 1)) ** eta))
                except:
                    pass
            m = max(m, eps)
            w[slc] = np.sqrt(m).astype(np.float32)
        return w, zero_mask

    def fit(self, basis_data: np.ndarray, ridge_train_data: np.ndarray,
            val_data: np.ndarray, horizons: List[int]):
        print("  [STD] Learning multi-scale seasonal bases...")
        for bs, dec in self.std_decomposers.items():
            comp = dec.decompose(basis_data)
            if comp is not None:
                dec.fit_seasonal_basis(comp)
                print(f"    - Scale {bs}: basis learned (rank={dec.seasonal_rank})")

        self._infer_feature_dim_and_groups(ridge_train_data[0, :self.lookback],
                                          self._extract_features_single)
        print(f"  Total features: {self.n_features_}")

        self._build_and_fit_model(ridge_train_data, val_data, horizons,
                                 extractor=self._extract_features_single,
                                 weight_maker=self._make_group_weight_vec_and_zero_mask)

# ======================= SSA Forecaster =======================
class SSANeuralForecaster(BaseForecaster):
    def __init__(self, lookback: int, block_sizes: List[int], seasonal_ranks: List[int],
                 group_multipliers: Optional[Dict[str, float]] = None):
        super().__init__()
        if len(block_sizes) != len(seasonal_ranks):
            raise ValueError("block_sizes and seasonal_ranks must have the same length")
        self.lookback = int(lookback)
        self.block_sizes = [int(b) for b in block_sizes]
        self.seasonal_ranks = [int(r) for r in seasonal_ranks]
        self.ssa_projects: Dict[int, SSAProjector] = {
            L: SSAProjector(L=L, seasonal_rank=sr, r_trend=1, max_samples=MAX_SAMPLES_SSA, seed=42+L)
            for L, sr in zip(self.block_sizes, self.seasonal_ranks)
        }

        cand = [L for L in self.block_sizes if L <= self.lookback]
        if 168 in cand:
            self._primary_L = 168
        elif len(cand) > 0:
            self._primary_L = max(cand)
        else:
            self._primary_L = self.block_sizes[0]

        self.group_multipliers_base = self._default_group_multipliers(group_multipliers)

    def _default_group_multipliers(self, user_override: Optional[Dict[str, float]]) -> Dict[str, float]:
        m = dict(DEFAULT_GROUP_MULTIPLIERS_BASE)
        m['slow_means'] = 0.6
        for L in self.block_sizes:
            m[f'ssa_L{L}_trend'] = m.get(f'ssa_L{L}_trend', 0.6)
            m[f'ssa_L{L}_seasonal'] = m.get(f'ssa_L{L}_seasonal', 1.0)
            m[f'ssa_L{L}_mean'] = m.get(f'ssa_L{L}_mean', 0.6)
            m[f'ssa_L{L}_energy'] = m.get(f'ssa_L{L}_energy', 0.9)
        if user_override:
            m.update(user_override)
        for k in list(m.keys()):
            if m[k] <= 0:
                m[k] = 1e-6
        return m

    def _extract_features_single(self, x_window_1d: np.ndarray, collect_slices: bool = False):
        pieces: List[np.ndarray] = []
        names: List[str] = []
        Lw = x_window_1d.shape[0]

        arr = x_window_1d.astype(np.float32, copy=False).ravel()
        pieces.append(arr); names.append('raw')

        for L, proj in self.ssa_projects.items():
            if Lw >= L and proj.U is not None and proj.r_actual > 0:
                coef, mu, energy = proj.project(x_window_1d)
                if coef is None:
                    continue
                rt = min(1, coef.shape[0])
                rs = max(0, coef.shape[0]-rt)
                if rt > 0:
                    pieces.append(coef[:rt].astype(np.float32, copy=False))
                    names.append(f'ssa_L{L}_trend')
                if rs > 0:
                    z_dir = (coef[rt:] / (energy if energy>0 else 1.0)).astype(np.float32, copy=False)
                    pieces.append(z_dir)
                    names.append(f'ssa_L{L}_seasonal')
                pieces.append(np.array([mu], dtype=np.float32))
                names.append(f'ssa_L{L}_mean')
                pieces.append(np.array([energy], dtype=np.float32))
                names.append(f'ssa_L{L}_energy')

        pL = self._primary_L
        if Lw >= pL:
            nb = max(1, Lw // pL)
            x_trim = x_window_1d[:nb*pL].reshape(nb, pL).astype(np.float32, copy=False)
            means = x_trim.mean(axis=1)
            if nb >= 2:
                t = np.arange(means.shape[0], dtype=np.float32)
                t_mean = t.mean(); y_mean = means.mean()
                denom = np.sum((t - t_mean) ** 2) + 1e-8
                slope = np.sum((t - t_mean) * (means - y_mean)) / denom
            else:
                slope = 0.0
            pieces.append(np.array([slope], dtype=np.float32))
            names.append('trend_dyn')
            K = min(self.slow_means_K, means.shape[0])
            if K > 0:
                pieces.append(means[-K:].astype(np.float32))
                names.append('slow_means')

        xw = x_window_1d.astype(np.float32, copy=False)
        q25 = np.percentile(xw, 25).astype(np.float32)
        q75 = np.percentile(xw, 75).astype(np.float32)
        pieces.append(np.array([xw.mean(), xw.std(), xw.min(), xw.max(), q25, q75], dtype=np.float32))
        names.append('stats')

        vec = np.concatenate(pieces, dtype=np.float32)
        if collect_slices:
            self.group_slices.clear()
            pos = 0
            for piece, name in zip(pieces, names):
                n = piece.shape[0]
                self.group_slices[name] = slice(pos, pos + n)
                pos += n
        return vec

    def _make_group_weight_vec_and_zero_mask(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        w = np.ones((self.n_features_,), dtype=np.float32)
        zero_mask = np.zeros((self.n_features_,), dtype=bool)
        eps = 1e-8
        for g, slc in self.group_slices.items():
            m = self.group_multipliers_base.get(g, 1.0)
            if g == 'raw':
                u = horizon / max(self._primary_L, 1)
                m = m * (1.0 + 0.3 * max(0.0, u - 1.0))
            elif g.startswith('ssa_L') and g.endswith('_seasonal'):
                try:
                    L = int(g.split('_')[0].replace('ssa_L',''))
                    u = horizon / max(L, 1)
                    if u >= 2.0:
                        zero_mask[slc] = True
                        m = m * 1e3
                    else:
                        m = m * (1.0 + (max(0.0, u - 1.0))**2 * 8.0)
                except:
                    m = m * 2.0
            elif g == 'ssa_L24_trend':
                u = horizon / 24.0
                if u >= 14.0:
                    zero_mask[slc] = True
                    m = m * 1e3
                else:
                    m = m * (1.0 + (max(0.0, u - 7.0))**2 * 2.0)
            elif g in (f'ssa_L{self._primary_L}_trend', f'ssa_L{self._primary_L}_mean', 'slow_means'):
                m = m * 0.7
            elif g.endswith('_trend') or g.endswith('_mean'):
                m = m * 0.85
            elif g.endswith('_energy'):
                m = m * 0.9
            m = max(m, eps)
            w[slc] = np.sqrt(m).astype(np.float32)
        return w, zero_mask

    def fit(self, basis_data: np.ndarray, ridge_train_data: np.ndarray,
            val_data: np.ndarray, horizons: List[int]):
        print("  [SSA] Learning multi-scale SSA bases...")
        for L, proj in self.ssa_projects.items():
            proj.fit_from_series(basis_data)
            print(f"    - SSA L={L}: basis learned (r_total={proj.r_total}, r_actual={proj.r_actual})")

        self._infer_feature_dim_and_groups(ridge_train_data[0, :self.lookback],
                                          self._extract_features_single)
        print(f"  Total features: {self.n_features_}")

        self._build_and_fit_model(ridge_train_data, val_data, horizons,
                                 extractor=self._extract_features_single,
                                 weight_maker=self._make_group_weight_vec_and_zero_mask)

# ======================= Dataset-specific patch =======================
def _canon_key(dataset_name: str) -> str:
    s = dataset_name.lower()
    if 'etth1' in s: return 'etth1'
    if 'etth2' in s: return 'etth2'
    if 'ettm1' in s: return 'ettm1'
    if 'ettm2' in s: return 'ettm2'
    if 'electricity' in s: return 'electricity'
    if 'traffic' in s: return 'traffic'
    if 'weather' in s: return 'weather'
    if 'exchange' in s: return 'exchange_rate'
    return 'etth1'

DATASET_TUNING = {
    'etth1': {'lookback': 336, 'primary_L': 168, 'seasonal_gate_ratio': 2.0,
              'raw_slope': 0.3, 'slow_means_K': 4, 'gate_24h_trend_after_weeks': 2, 'seasonal_rank_scale': 1.0},
    'etth2': {'lookback': 336, 'primary_L': 168, 'seasonal_gate_ratio': 2.0,
              'raw_slope': 0.3, 'slow_means_K': 4, 'gate_24h_trend_after_weeks': 2, 'seasonal_rank_scale': 1.0},
    'ettm1': {'lookback': 672, 'primary_L': 672, 'seasonal_gate_ratio': 2.0,
              'raw_slope': 0.25, 'slow_means_K': 3, 'gate_24h_trend_after_weeks': None, 'seasonal_rank_scale': 1.0},
    'ettm2': {'lookback': 672, 'primary_L': 672, 'seasonal_gate_ratio': 2.0,
              'raw_slope': 0.25, 'slow_means_K': 3, 'gate_24h_trend_after_weeks': None, 'seasonal_rank_scale': 1.0},
    'electricity': {'lookback': 336, 'primary_L': 168, 'seasonal_gate_ratio': 2.0,
                    'raw_slope': 0.2, 'slow_means_K': 4, 'gate_24h_trend_after_weeks': 2, 'seasonal_rank_scale': 1.0},
    'traffic': {'lookback': 336, 'primary_L': 168, 'seasonal_gate_ratio': 2.0,
                'raw_slope': 0.25, 'slow_means_K': 4, 'gate_24h_trend_after_weeks': 2, 'seasonal_rank_scale': 0.8},
    'weather': {'lookback': 336, 'primary_L': 168, 'seasonal_gate_ratio': 1.5,
                'raw_slope': 0.35, 'slow_means_K': 3, 'gate_24h_trend_after_weeks': 1, 'seasonal_rank_scale': 0.7},
    'exchange_rate': {'lookback': 96, 'primary_L': 30, 'seasonal_gate_ratio': 1.2,
                      'raw_slope': 0.1, 'slow_means_K': 2, 'gate_24h_trend_after_weeks': 0,
                      'seasonal_rank_scale': 0.5},
}

def resolve_ds_config(dataset_name: str, default_lookback: int,
                      default_block_sizes: List[int], default_ranks: List[int]):
    key = _canon_key(dataset_name)
    cfg = DATASET_TUNING.get(key, {})
    lookback = int(cfg.get('lookback', default_lookback))
    primary_L = cfg.get('primary_L', None)
    scale = float(cfg.get('seasonal_rank_scale', 1.0))
    ranks = [max(1, int(round(r * scale))) for r in default_ranks]
    ranks = [min(r, b) for r, b in zip(ranks, default_block_sizes)]
    return key, lookback, primary_L, ranks, cfg

def apply_dataset_patch(model, ds_key: str, cfg: dict, algo: str = 'SSA'):
    if hasattr(model, '_primary_L') and cfg.get('primary_L') is not None:
        model._primary_L = int(cfg['primary_L'])

    if hasattr(model, 'slow_means_K'):
        model.slow_means_K = int(cfg.get('slow_means_K', model.slow_means_K))
    else:
        setattr(model, 'slow_means_K', int(cfg.get('slow_means_K', 3)))

    gmb = getattr(model, 'group_multipliers_base', {})
    primary = getattr(model, '_primary_L', None)
    if primary:
        gmb[f'ssa_L{primary}_trend'] = min(gmb.get(f'ssa_L{primary}_trend', 0.8), 0.6)
        gmb[f'ssa_L{primary}_mean'] = min(gmb.get(f'ssa_L{primary}_mean', 0.8), 0.6)
    gmb['slow_means'] = min(gmb.get('slow_means', 0.8), 0.6)
    model.group_multipliers_base = gmb

    raw_slope = float(cfg.get('raw_slope', 0.3))
    gate_ratio = float(cfg.get('seasonal_gate_ratio', 2.0))
    gate_24w = cfg.get('gate_24h_trend_after_weeks', None)

    def _make_group_weight_vec_and_zero_mask_patched(self, horizon: int):
        n = self.n_features_
        w = np.ones((n,), dtype=np.float32)
        zero_mask = np.zeros((n,), dtype=bool)
        eps = 1e-8
        for g, slc in self.group_slices.items():
            m = self.group_multipliers_base.get(g, 1.0)
            if g == 'raw':
                denom = getattr(self, '_primary_L', None) or 168.0
                u = horizon / max(float(denom), 1.0)
                m = m * (1.0 + raw_slope * max(0.0, u - 1.0))
            elif re.match(r'(ssa|scale)_L?\d+_seasonal', g):
                mobj = re.search(r'(\d+)', g)
                L = float(mobj.group(1)) if mobj else 24.0
                u = horizon / max(L, 1.0)
                if u >= gate_ratio:
                    zero_mask[slc] = True
                    m = m * 1e3
                else:
                    delta = max(0.0, u - 1.0) / max(1e-6, gate_ratio - 1.0)
                    m = m * (1.0 + 6.0 * (delta**2))
            elif g in ('ssa_L24_trend', 'scale_24_trend') and gate_24w is not None:
                weeks = horizon / 168.0
                if weeks >= float(gate_24w):
                    zero_mask[slc] = True
                    m = m * 1e3
                else:
                    m = m * (1.0 + 2.0 * max(0.0, weeks - 1.0)**2)
            elif g in (f'ssa_L{getattr(self, "_primary_L", 168)}_trend',
                       f'ssa_L{getattr(self, "_primary_L", 168)}_mean', 'slow_means'):
                m = m * 0.7
            elif g.endswith('_trend') or g.endswith('_mean'):
                m = m * 0.85
            elif g.endswith('_energy'):
                m = m * 0.9
            m = max(m, eps)
            w[slc] = np.sqrt(m).astype(np.float32)
        return w, zero_mask

    model._make_group_weight_vec_and_zero_mask = MethodType(_make_group_weight_vec_and_zero_mask_patched, model)

# ======================= Evaluation =======================
def _normalize_scales(block_sizes_in, seasonal_ranks_in):
    bs = []
    seen = set()
    for b in block_sizes_in:
        b = int(b)
        if b > 0 and b not in seen:
            seen.add(b); bs.append(b)
    if seasonal_ranks_in is None or len(seasonal_ranks_in) == 0:
        rs = [4] * len(bs)
    else:
        rs = [int(r) for r in seasonal_ranks_in]
    if len(rs) == 1 and len(bs) > 1:
        rs = [rs[0]] * len(bs)
    elif len(rs) < len(bs):
        rs = rs + [rs[-1]] * (len(bs) - len(rs))
    elif len(rs) > len(bs):
        rs = rs[:len(bs)]
    for i in range(len(rs)):
        rs[i] = max(1, min(rs[i], bs[i]))
    return bs, rs

def evaluate_train_only(file_path: str, lookback_default: int = LOOKBACK_WINDOW,
                        horizons: List[int] = HORIZONS) -> Dict:
    dataset_name = os.path.basename(file_path)
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")

    key = next((k for k in DATASET_PARAMS if k in dataset_name.lower()), 'etth')
    params = DATASET_PARAMS[key]
    block_sizes, seasonal_ranks = _normalize_scales(params['block_sizes'], params['seasonal_ranks'])

    ds_key, lookback, primary_L, tuned_ranks, cfg = resolve_ds_config(
        dataset_name, lookback_default, block_sizes, seasonal_ranks)
    seasonal_ranks = tuned_ranks
    print(f"Config: ds={ds_key} | lookback={lookback} | primary_L={primary_L}")

    Xraw, _ = load_csv(file_path)
    Xs, _, _ = zscore_channelwise(Xraw)
    ch, T = Xs.shape
    X_train, X_val, X_test, used_standard = split_by_standard(Xs, file_path)
    print(f"Data: {ch} channels × {T} timesteps")
    print(f"Split: {'standard' if used_standard else 'ratio'}")
    print(f"Protocol: BASIS on {LEARN_BASES_ON.upper()}, Model on TRAIN")

    if LEARN_BASES_ON == 'train':
        basis_data = X_train
    else:
        basis_data = np.concatenate([X_train, X_val], axis=1)

    if ALGO.upper() == 'STD':
        model = STDNeuralForecaster(lookback=lookback, block_sizes=block_sizes,
                                    seasonal_ranks=seasonal_ranks, group_multipliers=None)
    else:
        model = SSANeuralForecaster(lookback=lookback, block_sizes=block_sizes,
                                   seasonal_ranks=seasonal_ranks, group_multipliers=None)

    apply_dataset_patch(model, ds_key, cfg, algo=ALGO)

    model.fit(basis_data=basis_data, ridge_train_data=X_train,
             val_data=X_val, horizons=horizons)

    results = {}
    for horizon in horizons:
        print(f"\n  Predicting horizon {horizon}...")
        predictions = model.predict(X_test, horizon, extractor=model._extract_features_single)
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

    # Cleanup
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()

    return results

# ======================= Main =======================
def run_evaluation():
    print("="*80)
    print(f"{ALGO}-{MODEL_TYPE.upper()} Model - Server Optimized (In-Memory)")
    print("="*80)
    print(f"Device: {DEVICE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")
    print(f"Basis: {LEARN_BASES_ON.upper()} | Model: TRAIN with VAL early stopping")
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
                'algo': ALGO,
                'model_type': MODEL_TYPE
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
        output_file = f"{ALGO}_{MODEL_TYPE}_server_optimized_{timestamp}.csv"
        df_results.to_csv(output_file, index=False)

        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(df_results.to_string())
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    print("System Configuration:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA devices: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print()

    run_evaluation()
