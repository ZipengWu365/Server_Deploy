#!/usr/bin/env bash
set -e
CFG=$1
# Ensure package is installed/updated
pip install -e third_party/tsdecomp

# Run experiment
# Note: We need to run this as a module or install the main package.
# Since we added entry point to tsdecomp pyproject.toml but maybe not installed main repo as package?
# The prompt says "Wire setup.cfg/pyproject so decomp-linear-bench is importable".
# But decomp-linear-bench is in features/, which is part of the main repo, not tsdecomp.
# Wait, I added the script entry point to `third_party/tsdecomp/pyproject.toml`.
# That is incorrect if the code is in `features/decomp_linear_bench`.
# `features` is in the root of the repo. `tsdecomp` is in `third_party`.
# The CLI code `features.decomp_linear_bench.cli:main` is in the root repo.
# So I should probably run it via `python -m features.decomp_linear_bench.cli`.
# OR I need to make the root repo a package.
# Given the structure, running as module is safer.

python -m features.decomp_linear_bench.cli run --config ${CFG}

# Plot
# Use yq to get out_dir if available, else grep
OUT_DIR=$(grep "out_dir" ${CFG} | cut -d ":" -f 2 | tr -d " ")
# Trim whitespace
OUT_DIR=$(echo $OUT_DIR | xargs)

python -m features.decomp_linear_bench.cli plot --summary ${OUT_DIR}/metrics_summary_by_method_horizon_ablation.csv --out_dir ${OUT_DIR}
