# -*- coding: utf-8 -*-
"""
SERVER-OPTIMIZED RAW CNN (ABLATION STUDY) - IN-MEMORY PRE-EXTRACTION
Target Environment: High-memory (96GB+) Linux server with CUDA GPU.

This is an ablation study version, removing all SSA/STD feature engineering
to test a pure CNN model on raw time series windows.
"""

import os, gc
import psutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import time, json

# ============================ Server-Optimized Config ============================
# 🔑 Key parameters for server environment
DATA_DIR = "./dataset"
DATASETS = [
    "exchange_rate.csv","ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv",
    # "traffic.csv", "electricity.csv", "weather.csv",  # Removed temporarily
]
HORIZONS = [96, 192, 336, 720]

LOOKBACK_WINDOW = 336

# Algorithm and model selection
ALGO = 'RawCNN_Ablation'  # 仅支持 'RawCNN_Ablation'
LEARN_BASES_ON = 'train'  # 仅支持 'train'
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
            for ch_start in range(0, n_channels, 100): # Using 100 as chunk size
                ch_end = min(ch_start + 100, n_channels)
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
            
            # Use more workers on server (set to 0 for Windows to avoid multiprocessing issues)
            num_workers = 0 if os.name == 'nt' else min(os.cpu_count() // 2, 8)
            # Enable pin_memory for faster GPU transfer
            use_cuda = torch.cuda.is_available()
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=use_cuda)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=use_cuda)
            
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

    def fit(self, train_data: np.ndarray,
            val_data: np.ndarray, horizons: List[int]):
        
        print("  [RawCNN] Running Ablation Experiment (Raw Window -> CNN)")
        
        # 1. 学习特征维度 (现在 n_features_ 将等于 lookback)
        # 需要一个样本窗口来推断特征维度
        sample_window = train_data[0, :self.lookback]
        
        self._infer_feature_dim_and_groups(sample_window, # 将窗口直接传递
                                           self._extract_features_single)
        print(f"  [RawCNN] Total features (CNN input_dim): {self.n_features_}")

        if self.n_features_ != self.lookback:
             print(f"WARNING: Feature dim ({self.n_features_}) != Lookback ({self.lookback}). Forcing.")
             self.n_features_ = self.lookback # 强制设置

        # 2. 直接调用父类的模型训练
        self._build_and_fit_model(train_data, val_data, horizons,
                                  extractor=self._extract_features_single,
                                  weight_maker=self._make_group_weight_vec_and_zero_mask)

# ======================= Dataset-specific patch (Lookback only) =======================
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
    'etth1': {'lookback': 336},
    'etth2': {'lookback': 336},
    'ettm1': {'lookback': 672},
    'ettm2': {'lookback': 672},
    'electricity': {'lookback': 336},
    'traffic': {'lookback': 336},
    'weather': {'lookback': 336},
    'exchange_rate': {'lookback': 96},
}

def resolve_ds_config(dataset_name: str, default_lookback: int):
    key = _canon_key(dataset_name)
    cfg = DATASET_TUNING.get(key, {})
    lookback = int(cfg.get('lookback', default_lookback))
    return key, lookback, cfg

# ======================= Evaluation =======================
def evaluate_train_only(file_path: str, lookback_default: int = LOOKBACK_WINDOW,
                        horizons: List[int] = HORIZONS) -> Dict:
    dataset_name = os.path.basename(file_path)
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")

    ds_key, lookback, cfg = resolve_ds_config(
        dataset_name, lookback_default)
    
    print(f"Config: ds={ds_key} | lookback={lookback}")

    Xraw, _ = load_csv(file_path)
    Xs, _, _ = zscore_channelwise(Xraw) # 标准化
    ch, T = Xs.shape
    X_train, X_val, X_test, used_standard = split_by_standard(Xs, file_path)
    print(f"Data: {ch} channels × {T} timesteps")
    print(f"Split: {'standard' if used_standard else 'ratio'}")
    
    # 仅使用 'train'
    print(f"Protocol: Model on TRAIN")

    if ALGO.startswith('RawCNN'):
        print("--- RUNNING ABLATION: RawCNN (no features) ---")
        model = RawCNNForecaster(lookback=lookback)
    else:
        raise ValueError(f"This script is configured for RawCNN_Ablation, but ALGO is {ALGO}")

    model.fit(train_data=X_train,
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
    print(f"{ALGO}-{MODEL_TYPE.upper()} Model - Server Optimized (In-Memory) - ABLATION")
    print("="*80)
    print(f"Device: {DEVICE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")
    print(f"Model: TRAIN with VAL early stopping")
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
        output_file = f"{ALGO}_{MODEL_TYPE}_ablation_{timestamp}.csv"
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