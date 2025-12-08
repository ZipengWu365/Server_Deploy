import argparse
import yaml
import os
from .runner import run_experiment
from .report import generate_report

def main():
    parser = argparse.ArgumentParser(description="Decomp-Linear Benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run experiment")
    run_parser.add_argument("--config", required=True, help="Path to YAML config")
    
    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Generate report")
    plot_parser.add_argument("--summary", required=True, help="Path to summary CSV")
    plot_parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    
    args = parser.parse_args()
    
    if args.command == "run":
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        run_experiment(cfg)
        
    elif args.command == "plot":
        generate_report(args.summary, args.out_dir)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
