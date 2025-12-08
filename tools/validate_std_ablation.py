import argparse
import pandas as pd
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Reference CSV")
    parser.add_argument("--new", required=True, help="New run CSV")
    parser.add_argument("--metric", default="MAE", help="Metric to compare")
    parser.add_argument("--tol", type=float, default=0.01, help="Tolerance (1%)")
    args = parser.parse_args()
    
    ref_df = pd.read_csv(args.ref)
    new_df = pd.read_csv(args.new)
    
    # Join on keys: dataset, horizon, ablation, decomp (if present in ref)
    # Assuming ref might have different structure, but let's try to match common columns
    keys = ["dataset", "horizon", "ablation"]
    if "decomp" in ref_df.columns and "decomp" in new_df.columns:
        keys.append("decomp")
        
    merged = pd.merge(ref_df, new_df, on=keys, suffixes=("_ref", "_new"))
    
    if merged.empty:
        print("No matching keys found between ref and new CSVs.")
        sys.exit(1)
        
    metric_ref = f"{args.metric}_ref"
    metric_new = f"{args.metric}_new"
    
    if metric_ref not in merged.columns or metric_new not in merged.columns:
        print(f"Metric {args.metric} not found in columns.")
        sys.exit(1)
        
    # Compute delta
    # Avoid div by zero
    denom = merged[metric_ref].abs() + 1e-8
    delta = (merged[metric_new] - merged[metric_ref]).abs() / denom
    
    max_delta = delta.max()
    mean_delta = delta.mean()
    
    print(f"Max relative delta: {max_delta:.4f}")
    print(f"Mean relative delta: {mean_delta:.4f}")
    
    if max_delta > args.tol:
        print(f"FAILURE: Max delta {max_delta:.4f} exceeds tolerance {args.tol}")
        # Print violations
        violations = merged[delta > args.tol]
        print(violations[keys + [metric_ref, metric_new]])
        sys.exit(1)
    else:
        print("SUCCESS: Results match within tolerance.")
        sys.exit(0)

if __name__ == "__main__":
    main()
