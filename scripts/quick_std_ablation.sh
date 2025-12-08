#!/usr/bin/env bash
# Quick run with only STD methods
# Create a temp config
cat <<EOF > configs/decomp_linear/quick_std.yaml
dataset:
  name: ETTh1
  path: data/ETTh1.csv
lookback: 336
horizons: [96]
decomp: ["STD_MULTI","STD_FULL"]
ablation_modes: ["T","S","D","T+S+D"]
learner: "Ridge"
learner_params:
  alpha: 1.0
out_dir: outputs/decomp_linear_bench/quick_std/
EOF

./scripts/run_decomp_linear_bench.sh configs/decomp_linear/quick_std.yaml
