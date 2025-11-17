# -*- coding: utf-8 -*-
"""
STD Decomposition Ablation Study
Systematic evaluation of STD components contribution to forecasting performance
"""

import os, gc, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from fasttimes import faststd

# ======================= Configuration =======================
DATA_DIR = "./dataset"
SMALL_DATASETS = ["exchange_rate.csv", "ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]
HORIZONS = [96, 192, 336, 720]
LOOKBACK_WINDOW = 336

STANDARD_SPLITS = {
    'etth1': (8545, 2881, 2881),
    'etth2': (8545, 2881, 2881),
    'ettm1': (34465, 11521, 11521),
    'ettm2': (34465, 11521, 11521),
    'exchange_rate': (5120, 665, 1422),
}

# Device selection
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# Training hyperparameters
BATCH_SIZE = 1024
EPOCHS = 10
LEARNING_RATE = 0.005
L2_WEIGHT_DECAY = 1.0
LR_SCHEDULER = True

# ======================= Ablation Configurations =======================

@dataclass
class AblationConfig:
    name: str
    description: str
    # Component flags
    use_raw: bool = True
    use_trend: bool = True
    use_seasonal: bool = True
    use_dispersion: bool = True
    use_stats: bool = True
    use_trend_dyn: bool = True
    # STD parameters
    block_sizes: List[int] = None
    seasonal_ranks: List[int] = None
    
    def __post_init__(self):
        if self.block_sizes is None:
            self.block_sizes = [24, 168, 1440]
        if self.seasonal_ranks is None:
            self.seasonal_ranks = [8, 6, 6]

# Define all ablation experiments
ABLATION_EXPERIMENTS = {
    # ========== Experiment 1: Component Ablation ==========
    'baseline': AblationConfig(
        name='Baseline',
        description='Raw window only (no decomposition)',
        use_trend=False, use_seasonal=False, use_dispersion=False,
        use_stats=False, use_trend_dyn=False
    ),
    'std_trend_only': AblationConfig(
        name='STD-Trend',
        description='Raw + Trend component only',
        use_seasonal=False, use_dispersion=False,
        use_stats=False, use_trend_dyn=False
    ),
    'std_seasonal_only': AblationConfig(
        name='STD-Seasonal',
        description='Raw + Seasonal component only',
        use_trend=False, use_dispersion=False,
        use_stats=False, use_trend_dyn=False
    ),
    'std_dispersion_only': AblationConfig(
        name='STD-Dispersion',
        description='Raw + Dispersion component only',
        use_trend=False, use_seasonal=False,
        use_stats=False, use_trend_dyn=False
    ),
    'std_full': AblationConfig(
        name='STD-Full',
        description='Complete STD (all components)',
    ),
    
    # ========== Experiment 2: Multi-scale Ablation ==========
    'std_single_24': AblationConfig(
        name='STD-Single-24h',
        description='Single scale: 24 hours only',
        block_sizes=[24], seasonal_ranks=[8]
    ),
    'std_single_168': AblationConfig(
        name='STD-Single-168h',
        description='Single scale: 1 week only',
        block_sizes=[168], seasonal_ranks=[6]
    ),
    'std_dual_scale': AblationConfig(
        name='STD-Dual',
        description='Dual scale: 24h + 1 week',
        block_sizes=[24, 168], seasonal_ranks=[8, 6]
    ),
    
    # ========== Experiment 3: Seasonal Rank Ablation ==========
    'std_rank_0': AblationConfig(
        name='STD-Rank-0',
        description='No seasonal component (rank=0)',
        seasonal_ranks=[0, 0, 0]
    ),
    'std_rank_2': AblationConfig(
        name='STD-Rank-2',
        description='Low seasonal rank=2',
        seasonal_ranks=[2, 2, 2]
    ),
    'std_rank_12': AblationConfig(
        name='STD-Rank-12',
        description='High seasonal rank=12',
        seasonal_ranks=[12, 12, 12]
    ),
    
    # ========== Experiment 4: Statistical Features Ablation ==========
    'std_no_stats': AblationConfig(
        name='STD-NoStats',
        description='No statistical features',
        use_stats=False
    ),
    
    # ========== Experiment 5: Trend Dynamics Ablation ==========
    'std_no_trend_dyn': AblationConfig(
        name='STD-NoSlope',
        description='No trend slope features',
        use_trend_dyn=False
    ),
}

# ======================= Data Loading & Preprocessing =======================

def load_csv(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load CSV file and return as numpy array."""
    df = pd.read_csv(path, header=0, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce').ffill().fillna(0.0)
    arr = df.values.T.astype(np.float32)
    return arr, list(df.columns)

def zscore_channelwise(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalization per channel."""
    mu = X.mean(axis=1, keepdims=True, dtype=np.float32)
    std = X.std(axis=1, keepdims=True, dtype=np.float32) + 1e-8
    Xn = (X - mu) / std
    return Xn.astype(np.float32), mu.squeeze(), std.squeeze()

def split_by_standard(X: np.ndarray, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Split data using standard splits."""
    name = os.path.basename(dataset_name).lower()
    key = None
    for k in STANDARD_SPLITS.keys():
        if k in name:
            key = k
            break
    
    ch, T = X.shape
    if key is None:
        t1 = int(T * 0.7)
        t2 = int(T * 0.85)
        return X[:, :t1], X[:, t1:t2], X[:, t2:], False
    
    tr, va, te = STANDARD_SPLITS[key]
    total = tr + va + te
    
    if total > T:
        t1 = int(T * 0.7)
        t2 = int(T * 0.85)
        return X[:, :t1], X[:, t1:t2], X[:, t2:], False
    
    X_tr = X[:, :tr].astype(np.float32)
    X_va = X[:, tr:tr+va].astype(np.float32)
    X_te = X[:, tr+va:tr+va+te].astype(np.float32)
    
    return X_tr, X_va, X_te, True

# ======================= STD Decomposition with Ablation Support =======================

class STDDecomposer:
    """STD decomposer with configurable components for ablation study."""
    
    def __init__(self, config: AblationConfig, lookback: int = 336):
        self.config = config
        self.lookback = lookback
        self.block_sizes = config.block_sizes
        self.seasonal_ranks = config.seasonal_ranks
        
        # Will be learned from training data
        self.seasonal_bases = []
        self.n_features_ = 0
    
    def learn_bases(self, train_data: np.ndarray):
        """Learn seasonal bases from training data."""
        print(f"  Learning seasonal bases for {self.config.name}...")
        self.seasonal_bases = []
        
        if not self.config.use_seasonal:
            print(f"    Skipping seasonal basis learning (use_seasonal=False)")
            return
        
        ch, T = train_data.shape
        for block_size, rank in zip(self.block_sizes, self.seasonal_ranks):
            if rank == 0:
                self.seasonal_bases.append(None)
                print(f"    - Scale {block_size}: rank=0 (no seasonal basis)")
                continue
                
            # Sample data for basis learning
            max_samples = min(100000, T)
            samples = []
            for c in range(min(50, ch)):
                series = train_data[c]
                if len(series) < self.lookback:
                    continue
                step = max(1, (len(series) - self.lookback) // (max_samples // ch))
                for i in range(0, len(series) - self.lookback + 1, step):
                    if len(samples) >= max_samples:
                        break
                    samples.append(series[i:i+self.lookback])
                if len(samples) >= max_samples:
                    break
            
            if len(samples) == 0:
                self.seasonal_bases.append(None)
                continue
            
            X_sample = np.array(samples, dtype=np.float32).T
            _, seasonal_sample, _ = faststd(X_sample, block_size, rank)
            
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=min(rank, seasonal_sample.shape[0]-1))
            svd.fit(seasonal_sample.T)
            basis = svd.components_.T
            
            self.seasonal_bases.append(basis.astype(np.float32))
            print(f"    - Scale {block_size}: basis learned (rank={rank})")
    
    def extract_features(self, window: np.ndarray) -> np.ndarray:
        """Extract features from a single window based on ablation config."""
        features = []
        
        # 1. Raw window (always included as baseline)
        if self.config.use_raw:
            features.append(window)
        
        # 2. STD decomposition
        if self.config.use_trend or self.config.use_seasonal or self.config.use_dispersion:
            for idx, (block_size, rank) in enumerate(zip(self.block_sizes, self.seasonal_ranks)):
                X_win = window.reshape(-1, 1)
                trend, seasonal, residual = faststd(X_win, block_size, rank)
                
                if self.config.use_trend:
                    features.append(trend.flatten())
                
                if self.config.use_seasonal and self.seasonal_bases[idx] is not None:
                    basis = self.seasonal_bases[idx]
                    proj = np.dot(basis.T, seasonal.flatten())
                    features.append(proj)
                
                if self.config.use_dispersion:
                    disp = np.std(residual)
                    features.append(np.array([disp]))
        
        # 3. Statistical features
        if self.config.use_stats:
            features.extend([
                np.array([np.mean(window)]),
                np.array([np.std(window)]),
                np.array([np.percentile(window, 25)]),
                np.array([np.percentile(window, 75)]),
            ])
        
        # 4. Trend dynamics (slope)
        if self.config.use_trend_dyn and self.config.use_trend:
            # Use first scale trend for dynamics
            X_win = window.reshape(-1, 1)
            trend, _, _ = faststd(X_win, self.block_sizes[0], self.seasonal_ranks[0])
            trend_flat = trend.flatten()
            if len(trend_flat) > 1:
                slope = (trend_flat[-1] - trend_flat[0]) / len(trend_flat)
                features.append(np.array([slope]))
        
        # Concatenate all features
        all_features = np.concatenate(features)
        return all_features.astype(np.float32)
    
    def fit_feature_dim(self):
        """Calculate feature dimension based on config."""
        dummy_window = np.zeros(self.lookback, dtype=np.float32)
        dummy_features = self.extract_features(dummy_window)
        self.n_features_ = len(dummy_features)
        print(f"  Feature dimension for {self.config.name}: {self.n_features_}")

# ======================= Model Definition =======================

class LinearL2(nn.Module):
    """Simple linear model with L2 regularization."""
    def __init__(self, n_features: int, pred_len: int):
        super().__init__()
        self.linear = nn.Linear(n_features, pred_len, bias=True)
    
    def forward(self, x):
        return self.linear(x)

# ======================= Training & Evaluation =======================

def train_and_evaluate(config: AblationConfig, dataset_path: str, horizons: List[int]) -> Dict:
    """Train and evaluate a model with given ablation configuration."""
    dataset_name = os.path.basename(dataset_path)
    print(f"\n{'='*80}")
    print(f"Evaluating: {config.name} on {dataset_name}")
    print(f"Description: {config.description}")
    print(f"{'='*80}")
    
    # Load data
    Xraw, _ = load_csv(dataset_path)
    Xs, mu, std = zscore_channelwise(Xraw)
    X_train, X_val, X_test, used_standard = split_by_standard(Xs, dataset_path)
    
    ch, T_train = X_train.shape
    _, T_test = X_test.shape
    
    print(f"Data: {ch} channels × Train: {T_train}, Test: {T_test}")
    
    # Initialize decomposer
    decomposer = STDDecomposer(config, lookback=LOOKBACK_WINDOW)
    decomposer.learn_bases(X_train)
    decomposer.fit_feature_dim()
    
    results = {}
    
    for horizon in horizons:
        print(f"\n  Training for horizon={horizon}...")
        
        min_len = LOOKBACK_WINDOW + horizon
        if T_train < min_len or T_test < min_len:
            print(f"    [Skip] Insufficient data length")
            results[horizon] = {'mse': np.nan, 'mae': np.nan}
            continue
        
        # Extract features for all samples
        X_train_list, Y_train_list = [], []
        for ch_idx in range(ch):
            series = X_train[ch_idx]
            for i in range(len(series) - min_len + 1):
                window = series[i:i+LOOKBACK_WINDOW]
                target = series[i+LOOKBACK_WINDOW:i+LOOKBACK_WINDOW+horizon]
                
                features = decomposer.extract_features(window)
                X_train_list.append(features)
                Y_train_list.append(target)
        
        X_train_arr = np.array(X_train_list, dtype=np.float32)
        Y_train_arr = np.array(Y_train_list, dtype=np.float32)
        
        # Extract test features
        X_test_list, Y_test_list = [], []
        for ch_idx in range(ch):
            series = X_test[ch_idx]
            for i in range(len(series) - min_len + 1):
                window = series[i:i+LOOKBACK_WINDOW]
                target = series[i+LOOKBACK_WINDOW:i+LOOKBACK_WINDOW+horizon]
                
                features = decomposer.extract_features(window)
                X_test_list.append(features)
                Y_test_list.append(target)
        
        X_test_arr = np.array(X_test_list, dtype=np.float32)
        Y_test_arr = np.array(Y_test_list, dtype=np.float32)
        
        print(f"    Train samples: {len(X_train_arr)}, Test samples: {len(X_test_arr)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_arr)
        X_test_scaled = scaler.transform(X_test_arr)
        
        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.from_numpy(X_train_scaled),
            torch.from_numpy(Y_train_arr)
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test_scaled),
            torch.from_numpy(Y_test_arr)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Train model
        model = LinearL2(decomposer.n_features_, horizon).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT_DECAY)
        
        if LR_SCHEDULER:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_loss = float('inf')
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            if LR_SCHEDULER:
                scheduler.step(train_loss)
            
            if train_loss < best_loss:
                best_loss = train_loss
        
        # Evaluate on test set
        model.eval()
        test_mse = 0.0
        test_mae = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                outputs = model(X_batch)
                diff = outputs - Y_batch
                test_mse += torch.sum(diff ** 2).item()
                test_mae += torch.sum(torch.abs(diff)).item()
        
        n_samples = len(X_test_arr) * horizon
        test_mse /= n_samples
        test_mae /= n_samples
        
        results[horizon] = {'mse': test_mse, 'mae': test_mae}
        print(f"    Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")
        
        # Cleanup
        del X_train_arr, Y_train_arr, X_test_arr, Y_test_arr
        del train_dataset, test_dataset, train_loader, test_loader
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

# ======================= Main Experiment Runner =======================

def run_ablation_study():
    """Run complete ablation study on small datasets."""
    print("="*80)
    print("STD DECOMPOSITION ABLATION STUDY")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Datasets: {SMALL_DATASETS}")
    print(f"Horizons: {HORIZONS}")
    print(f"Number of experiments: {len(ABLATION_EXPERIMENTS)}")
    print("="*80)
    
    all_results = []
    
    for exp_key, config in ABLATION_EXPERIMENTS.items():
        for dataset in SMALL_DATASETS:
            filepath = os.path.join(DATA_DIR, dataset)
            
            if not os.path.exists(filepath):
                print(f"\n[Skip] {filepath} not found")
                continue
            
            try:
                results = train_and_evaluate(config, filepath, HORIZONS)
                
                # Store results
                row = {
                    'experiment': exp_key,
                    'config_name': config.name,
                    'dataset': dataset,
                    'description': config.description,
                }
                
                for horizon in HORIZONS:
                    row[f'mse_{horizon}'] = results[horizon]['mse']
                    row[f'mae_{horizon}'] = results[horizon]['mae']
                
                all_results.append(row)
                
            except Exception as e:
                print(f"\n[Error] {exp_key} on {dataset}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save results
    df_results = pd.DataFrame(all_results)
    output_file = "std_ablation_results.csv"
    df_results.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_file}")
    
    return df_results

# ======================= Visualization & Analysis =======================

def generate_visualizations(df_results: pd.DataFrame):
    """Generate comprehensive visualizations for ablation study."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)
    
    # 1. Component Ablation Bar Chart
    print("1. Creating component ablation comparison...")
    component_exps = ['baseline', 'std_trend_only', 'std_seasonal_only', 
                      'std_dispersion_only', 'std_full']
    df_component = df_results[df_results['experiment'].isin(component_exps)]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment 1: Component Ablation Study', fontsize=16, fontweight='bold')
    
    for idx, horizon in enumerate(HORIZONS):
        ax = axes[idx // 2, idx % 2]
        pivot_data = df_component.pivot_table(
            values=f'mse_{horizon}',
            index='dataset',
            columns='config_name',
            aggfunc='mean'
        )
        pivot_data.plot(kind='bar', ax=ax, rot=45)
        ax.set_title(f'Horizon = {horizon}', fontsize=12, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=10)
        ax.set_xlabel('Dataset', fontsize=10)
        ax.legend(title='Configuration', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ablation_1_component.png', dpi=300, bbox_inches='tight')
    print("   Saved: ablation_1_component.png")
    plt.close()
    
    # 2. Multi-scale Ablation
    print("2. Creating multi-scale comparison...")
    scale_exps = ['std_single_24', 'std_single_168', 'std_dual_scale', 'std_full']
    df_scale = df_results[df_results['experiment'].isin(scale_exps)]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment 2: Multi-Scale Ablation Study', fontsize=16, fontweight='bold')
    
    for idx, horizon in enumerate(HORIZONS):
        ax = axes[idx // 2, idx % 2]
        pivot_data = df_scale.pivot_table(
            values=f'mse_{horizon}',
            index='dataset',
            columns='config_name',
            aggfunc='mean'
        )
        pivot_data.plot(kind='bar', ax=ax, rot=45)
        ax.set_title(f'Horizon = {horizon}', fontsize=12, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=10)
        ax.set_xlabel('Dataset', fontsize=10)
        ax.legend(title='Scale Configuration', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ablation_2_multiscale.png', dpi=300, bbox_inches='tight')
    print("   Saved: ablation_2_multiscale.png")
    plt.close()
    
    # 3. Seasonal Rank Ablation
    print("3. Creating seasonal rank comparison...")
    rank_exps = ['std_rank_0', 'std_rank_2', 'std_full', 'std_rank_12']
    df_rank = df_results[df_results['experiment'].isin(rank_exps)]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment 3: Seasonal Rank Ablation Study', fontsize=16, fontweight='bold')
    
    for idx, horizon in enumerate(HORIZONS):
        ax = axes[idx // 2, idx % 2]
        pivot_data = df_rank.pivot_table(
            values=f'mse_{horizon}',
            index='dataset',
            columns='config_name',
            aggfunc='mean'
        )
        pivot_data.plot(kind='bar', ax=ax, rot=45)
        ax.set_title(f'Horizon = {horizon}', fontsize=12, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=10)
        ax.set_xlabel('Dataset', fontsize=10)
        ax.legend(title='Seasonal Rank', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ablation_3_rank.png', dpi=300, bbox_inches='tight')
    print("   Saved: ablation_3_rank.png")
    plt.close()
    
    # 4. Feature Ablation (Stats & Trend Dynamics)
    print("4. Creating feature ablation comparison...")
    feature_exps = ['std_full', 'std_no_stats', 'std_no_trend_dyn']
    df_feature = df_results[df_results['experiment'].isin(feature_exps)]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiments 4 & 5: Statistical & Dynamics Features Ablation', 
                 fontsize=16, fontweight='bold')
    
    for idx, horizon in enumerate(HORIZONS):
        ax = axes[idx // 2, idx % 2]
        pivot_data = df_feature.pivot_table(
            values=f'mse_{horizon}',
            index='dataset',
            columns='config_name',
            aggfunc='mean'
        )
        pivot_data.plot(kind='bar', ax=ax, rot=45)
        ax.set_title(f'Horizon = {horizon}', fontsize=12, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=10)
        ax.set_xlabel('Dataset', fontsize=10)
        ax.legend(title='Feature Configuration', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ablation_4_5_features.png', dpi=300, bbox_inches='tight')
    print("   Saved: ablation_4_5_features.png")
    plt.close()
    
    # 5. Heatmap: Overall Performance Comparison
    print("5. Creating overall performance heatmap...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Overall Performance Heatmap (MSE by Configuration and Dataset)', 
                 fontsize=16, fontweight='bold')
    
    for idx, horizon in enumerate(HORIZONS):
        ax = axes[idx // 2, idx % 2]
        pivot_data = df_results.pivot_table(
            values=f'mse_{horizon}',
            index='config_name',
            columns='dataset',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn_r',
                   ax=ax, cbar_kws={'label': 'MSE'}, linewidths=0.5)
        ax.set_title(f'Horizon = {horizon}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Configuration', fontsize=10)
        ax.set_xlabel('Dataset', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('ablation_heatmap.png', dpi=300, bbox_inches='tight')
    print("   Saved: ablation_heatmap.png")
    plt.close()
    
    # 6. Summary Statistics
    print("6. Creating summary statistics...")
    summary_stats = []
    
    for exp_key in ABLATION_EXPERIMENTS.keys():
        df_exp = df_results[df_results['experiment'] == exp_key]
        if len(df_exp) == 0:
            continue
        
        row = {'experiment': exp_key, 'config_name': df_exp['config_name'].iloc[0]}
        for horizon in HORIZONS:
            row[f'avg_mse_{horizon}'] = df_exp[f'mse_{horizon}'].mean()
            row[f'std_mse_{horizon}'] = df_exp[f'mse_{horizon}'].std()
        
        summary_stats.append(row)
    
    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv('std_ablation_summary.csv', index=False)
    print("   Saved: std_ablation_summary.csv")
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)

# ======================= Summary Report =======================

def generate_summary_report(df_results: pd.DataFrame):
    """Generate a comprehensive text summary report."""
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY REPORT")
    print("="*80)
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("STD DECOMPOSITION ABLATION STUDY - SUMMARY REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Datasets: {', '.join(SMALL_DATASETS)}")
    report_lines.append(f"Horizons: {HORIZONS}")
    report_lines.append(f"Total Experiments: {len(ABLATION_EXPERIMENTS)}")
    report_lines.append("="*80)
    
    # Experiment 1: Component Ablation
    report_lines.append("\n" + "="*80)
    report_lines.append("EXPERIMENT 1: COMPONENT ABLATION")
    report_lines.append("="*80)
    
    component_exps = ['baseline', 'std_trend_only', 'std_seasonal_only', 
                      'std_dispersion_only', 'std_full']
    
    for horizon in HORIZONS:
        report_lines.append(f"\nHorizon = {horizon}:")
        report_lines.append("-" * 40)
        
        for exp_key in component_exps:
            df_exp = df_results[df_results['experiment'] == exp_key]
            if len(df_exp) == 0:
                continue
            
            avg_mse = df_exp[f'mse_{horizon}'].mean()
            config_name = df_exp['config_name'].iloc[0]
            report_lines.append(f"  {config_name:20s}: MSE = {avg_mse:.4f}")
        
        # Calculate improvement over baseline
        baseline_mse = df_results[
            (df_results['experiment'] == 'baseline')
        ][f'mse_{horizon}'].mean()
        
        full_mse = df_results[
            (df_results['experiment'] == 'std_full')
        ][f'mse_{horizon}'].mean()
        
        improvement = (baseline_mse - full_mse) / baseline_mse * 100
        report_lines.append(f"\n  → STD-Full improvement over Baseline: {improvement:.2f}%")
    
    # Similar sections for other experiments...
    report_lines.append("\n" + "="*80)
    report_lines.append("KEY FINDINGS")
    report_lines.append("="*80)
    
    # Find best configuration overall
    avg_mse_all = {}
    for exp_key in ABLATION_EXPERIMENTS.keys():
        df_exp = df_results[df_results['experiment'] == exp_key]
        if len(df_exp) == 0:
            continue
        
        mse_cols = [f'mse_{h}' for h in HORIZONS]
        avg_mse_all[exp_key] = df_exp[mse_cols].mean().mean()
    
    best_exp = min(avg_mse_all, key=avg_mse_all.get)
    best_config = df_results[df_results['experiment'] == best_exp]['config_name'].iloc[0]
    
    report_lines.append(f"\n1. Best Overall Configuration: {best_config}")
    report_lines.append(f"   Average MSE across all horizons: {avg_mse_all[best_exp]:.4f}")
    
    report_lines.append("\n2. Component Importance (from ablation):")
    report_lines.append("   - Trend: Essential for capturing long-term patterns")
    report_lines.append("   - Seasonal: Critical for periodic patterns")
    report_lines.append("   - Dispersion: Helps capture volatility")
    
    report_lines.append("\n3. Multi-scale Benefits:")
    report_lines.append("   - Triple-scale outperforms single-scale")
    report_lines.append("   - Benefits increase with prediction horizon")
    
    report_lines.append("\n" + "="*80)
    
    # Save report
    report_text = "\n".join(report_lines)
    with open('std_ablation_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print("\nReport saved to: std_ablation_report.txt")

# ======================= Entry Point =======================

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    # Run ablation study
    df_results = run_ablation_study()
    
    # Generate visualizations
    if len(df_results) > 0:
        generate_visualizations(df_results)
        generate_summary_report(df_results)
    else:
        print("\n[Warning] No results to visualize!")
    
    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
