# -*- coding: utf-8 -*-
"""
SERVER-OPTIMIZED 1D CNN - IN-MEMORY PRE-EXTRACTION VERSION
Target Environment: High-memory (96GB+) Linux server with CUDA GPU.

Optimizations:

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
    "exchange_rate.csv", "ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv",
    # "traffic.csv", "electricity.csv", "weather.csv",  # Removed temporarily
]
HORIZONS = [96, 192, 336, 720]

LOOKBACK_WINDOW = 336

# Algorithm and model selection
ALGO = 'RawCNN_Ablation'  # 'STD', 'SSA', or 'RawCNN_Ablation' for ablation study
LEARN_BASES_ON = 'train'  # 只支持 'train'
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
EPOCHS = 10            # Reduced for quick testing
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


# ======================= Base Forecaster Class =======================
class BaseForecaster:
    """Base class for forecasters"""
    def __init__(self):
        self.n_features_ = None
        self.group_slices = {}
        self.models = {}  # {horizon: trained_model}
        self.group_multipliers_base = DEFAULT_GROUP_MULTIPLIERS_BASE.copy()
    
    def _infer_feature_dim_and_groups(self, sample_window, extractor):
        """Infer feature dimension from a sample window"""
        feats = extractor(sample_window, collect_slices=True)
        self.n_features_ = feats.shape[0]
    
    def _build_and_fit_model(self, train_data, val_data, horizons, extractor, weight_maker):
        """Build and train models for each horizon"""
        ch_train, T_train = train_data.shape
        ch_val, T_val = val_data.shape
        
        for horizon in horizons:
            print(f"\n  Training CNN model, horizon {horizon}...")
            print_memory(f"Before training H={horizon}")
            
            # ========== Phase 1: Pre-extract ALL training features to RAM ==========
            print(f"    Phase 1: Pre-extracting all training features to RAM...")
            n_train = max(0, T_train - (self.lookback + horizon) + 1)
            X_train_all = np.empty((ch_train * n_train, self.n_features_), dtype=np.float32)
            Y_train_all = np.empty((ch_train * n_train, horizon), dtype=np.float32)
            
            idx = 0
            for ch_idx in range(ch_train):
                series = train_data[ch_idx]
                for i in range(n_train):
                    win = series[i: i + self.lookback]
                    X_train_all[idx] = extractor(win)
                    Y_train_all[idx] = series[i + self.lookback: i + self.lookback + horizon]
                    idx += 1
            print(f"    Total train samples: {idx}")
            print_memory("After train feature extraction to RAM")
            
            # ========== Phase 2: Extract validation features ==========
            print(f"    Phase 2: Extracting validation features...")
            n_val = max(0, T_val - (self.lookback + horizon) + 1)
            
            # If validation set is too short, use a portion of training data
            if n_val == 0 or ch_val * n_val < 100:
                print(f"    ⚠ Val set too short for H={horizon}, using a part of train data for validation.")
                n_val_from_train = min(n_train // 10, 5000 // ch_train)  # Use 10% or up to 5000 samples
                X_val = np.empty((ch_train * n_val_from_train, self.n_features_), dtype=np.float32)
                Y_val = np.empty((ch_train * n_val_from_train, horizon), dtype=np.float32)
                
                idx_val = 0
                for ch_idx in range(ch_train):
                    series = train_data[ch_idx]
                    for i in range(n_train - n_val_from_train, n_train):
                        win = series[i: i + self.lookback]
                        X_val[idx_val] = extractor(win)
                        Y_val[idx_val] = series[i + self.lookback: i + self.lookback + horizon]
                        idx_val += 1
            else:
                X_val = np.empty((ch_val * n_val, self.n_features_), dtype=np.float32)
                Y_val = np.empty((ch_val * n_val, horizon), dtype=np.float32)
                
                idx_val = 0
                for ch_idx in range(ch_val):
                    series = val_data[ch_idx]
                    for i in range(n_val):
                        win = series[i: i + self.lookback]
                        X_val[idx_val] = extractor(win)
                        Y_val[idx_val] = series[i + self.lookback: i + self.lookback + horizon]
                        idx_val += 1
            print(f"    Val samples: {idx_val}")
            print_memory("After validation extraction")
            
            # ========== Phase 3: Fit Scaler and create DataLoaders ==========
            print("    Phase 3: Fitting scaler and creating DataLoaders...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_all).astype(np.float32)
            X_val_scaled = scaler.transform(X_val).astype(np.float32)
            
            train_dataset = TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(Y_train_all))
            val_dataset = TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(Y_val))
            
            # Use more workers on server (set to 0 for Windows to avoid multiprocessing issues)
            num_workers = 0 if os.name == 'nt' else min(os.cpu_count() // 2, 8)
            # Enable pin_memory for faster GPU transfer
            use_cuda = torch.cuda.is_available()
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=use_cuda)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=use_cuda)
            
            del X_train_all, Y_train_all, X_train_scaled, X_val, Y_val, X_val_scaled
            gc.collect()
            print_memory("After DataLoader creation")
            
            # ========== Phase 4: Train model ==========
            print("    Phase 4: Starting model training...")
            model = CNN1D(self.n_features_, horizon).to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT_DECAY)
            if LR_SCHEDULER:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(EPOCHS):
                model.train()
                epoch_loss = 0.0
                for X_batch, Y_batch in train_loader:
                    X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = nn.MSELoss()(outputs, Y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                epoch_train_loss = epoch_loss / len(train_loader)
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, Y_batch in val_loader:
                        X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                        outputs = model(X_batch)
                        loss = nn.MSELoss()(outputs, Y_batch)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                
                if LR_SCHEDULER:
                    scheduler.step(val_loss)
                
                if (epoch + 1) % 5 == 0:
                    print(f"      Epoch {epoch+1:3d}/{EPOCHS} | Train: {epoch_train_loss:.4f} | Val: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.models[horizon] = model
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"      Early stopping at epoch {epoch+1}")
                        break
            
            print(f"    ✓ Training complete | Best val loss: {best_val_loss:.4f}")
            print_memory(f"After H={horizon} training")
            
            del train_loader, val_loader, train_dataset, val_dataset
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
    
    def predict(self, test_data, horizon, extractor):
        """Make predictions on test data"""
        model = self.models.get(horizon)
        if model is None:
            return {}
        
        model.eval()
        ch_test, T_test = test_data.shape
        predictions = {}
        
        with torch.no_grad():
            for ch_idx in range(ch_test):
                series = test_data[ch_idx]
                n_test = max(0, T_test - (self.lookback + horizon) + 1)
                if n_test <= 0:
                    predictions[ch_idx] = np.array([])
                    continue
                
                X_test = np.empty((n_test, self.n_features_), dtype=np.float32)
                for i in range(n_test):
                    win = series[i: i + self.lookback]
                    X_test[i] = extractor(win)
                
                X_test_tensor = torch.from_numpy(X_test).to(DEVICE)
                Y_pred = model(X_test_tensor).cpu().numpy()
                predictions[ch_idx] = Y_pred
        
        return predictions

# ======================= RAW CNN (Ablation) =======================
class RawCNNForecaster(BaseForecaster):
    """
    消融实验版本：只使用原始窗口数据作为CNN的输入。
    移除了所有SSA/STD特征工程。
    """
    def __init__(self, lookback: int):
        super().__init__()
        self.lookback = int(lookback)
        global MODEL_TYPE
        if MODEL_TYPE != 'cnn':
            print(f"WARN: RawCNNForecaster requires MODEL_TYPE='cnn'. Forcing it.")
            MODEL_TYPE = 'cnn'

    def _extract_features_single(self, x_window_1d: np.ndarray, collect_slices: bool = False):
        """
        消融版特征提取器：只返回原始窗口。
        """
        if collect_slices:
            self.group_slices.clear()
            self.group_slices['raw'] = slice(0, x_window_1d.shape[0])
        
        # 直接返回原始数据向量
        return x_window_1d.astype(np.float32, copy=False).ravel()

    def _make_group_weight_vec_and_zero_mask(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        消融版权重：所有权重为1，没有mask。
        """
        if self.n_features_ is None:
             self.n_features_ = self.lookback

        w = np.ones((self.n_features_,), dtype=np.float32)
        zero_mask = np.zeros((self.n_features_,), dtype=bool)
        return w, zero_mask

    def fit(self, basis_data: np.ndarray, ridge_train_data: np.ndarray,
            val_data: np.ndarray, horizons: List[int]):
        
        print("  [RawCNN] Running Ablation Experiment (Raw Window -> CNN)")
        
        # 1. 学习特征维度 (现在 n_features_ 将等于 lookback)
        # (basis_data is ignored)
        
        # 需要一个样本窗口来推断特征维度
        sample_window = ridge_train_data[0, :self.lookback]
        
        self._infer_feature_dim_and_groups(sample_window, # 将窗口直接传递
                                           self._extract_features_single)
        print(f"  [RawCNN] Total features (CNN input_dim): {self.n_features_}")

        if self.n_features_ != self.lookback:
             print(f"WARNING: Feature dim ({self.n_features_}) != Lookback ({self.lookback}). Forcing.")
             self.n_features_ = self.lookback # 强制设置

        # 2. 直接调用父类的模型训练
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
    
    # 仅使用 'train'
    basis_data = X_train
    print(f"Protocol: Model on TRAIN")


    if ALGO.startswith('RawCNN'):
        print("--- RUNNING ABLATION: RawCNN (no SSA/STD features) ---")
        model = RawCNNForecaster(lookback=lookback)
        # 不对 RawCNN 应用 apply_dataset_patch


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
    print(f"Basis: {LEARN_BASES_ON.upper()} (N/A for RawCNN) | Model: TRAIN with VAL early stopping")
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