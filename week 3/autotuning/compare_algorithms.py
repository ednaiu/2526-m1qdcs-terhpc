#!/usr/bin/env python3
"""
compare_algorithms.py — Head-to-head comparison of optimization algorithms

Runs all 4 optimization algorithms on the same problem with the same evaluation
budget, and compares:
    - Best GFLOP/s found
    - Number of evaluations to reach best
    - Wall time
    - Convergence curves

Algorithms:
    1. Gradient Descent (with random restarts)
    2. Bayesian Optimization (GP + Expected Improvement)
    3. Simulated Annealing
    4. Random Search (baseline)

Usage:
    python3 compare_algorithms.py --M 1024 --threads 4
    python3 compare_algorithms.py --M 512 --budget 30 -v
"""

import os
import sys
import json
import argparse
import time
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_runner import BenchmarkRunner, SGEMMConfig
from gradient_descent import GradientDescentOptimizer, OptimizationResult
from bayesian_opt import BayesianOptimizer
from simulated_annealing import SimulatedAnnealingOptimizer
from random_search import RandomSearchOptimizer

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def compute_convergence(history: list) -> list:
    """Compute running best GFLOP/s from evaluation history."""
    best = 0.0
    curve = []
    for h in history:
        g = h.get('gflops', 0.0)
        if g > best:
            best = g
        curve.append(best)
    return curve


def run_comparison(M: int, N: int, K: int, kernel: str, threads: int,
                   budget: int, verbose: bool) -> dict:
    """Run all algorithms and return comparison results."""

    runner = BenchmarkRunner(verbose=verbose)
    results = {}

    # 1. Gradient Descent
    print(f"\n{'='*60}")
    print("Algorithm 1/4: Gradient Descent")
    print(f"{'='*60}")
    runner.clear_cache()
    gd_opt = GradientDescentOptimizer(
        runner, M, N, K, kernel=kernel, threads=threads, verbose=verbose
    )
    gd_result = gd_opt.optimize_with_restarts(
        n_restarts=min(3, max(1, budget // 15)),
        max_iterations=max(5, budget // 3)
    )
    results['gradient_descent'] = gd_result

    # 2. Bayesian Optimization
    print(f"\n{'='*60}")
    print("Algorithm 2/4: Bayesian Optimization")
    print(f"{'='*60}")
    runner.clear_cache()
    bo_opt = BayesianOptimizer(
        runner, M, N, K, kernel=kernel, threads=threads, verbose=verbose
    )
    bo_result = bo_opt.optimize(
        n_iterations=budget,
        n_init=min(5, budget // 3)
    )
    results['bayesian_optimization'] = bo_result

    # 3. Simulated Annealing
    print(f"\n{'='*60}")
    print("Algorithm 3/4: Simulated Annealing")
    print(f"{'='*60}")
    runner.clear_cache()
    sa_opt = SimulatedAnnealingOptimizer(
        runner, M, N, K, kernel=kernel, threads=threads, verbose=verbose
    )
    sa_result = sa_opt.optimize(n_iterations=budget)
    results['simulated_annealing'] = sa_result

    # 4. Random Search (baseline)
    print(f"\n{'='*60}")
    print("Algorithm 4/4: Random Search (baseline)")
    print(f"{'='*60}")
    runner.clear_cache()
    rs_opt = RandomSearchOptimizer(
        runner, M, N, K, kernel=kernel, threads=threads, verbose=verbose
    )
    rs_result = rs_opt.optimize(n_iterations=budget)
    results['random_search'] = rs_result

    return results


def print_summary(results: dict, M: int, N: int, K: int):
    """Print formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"ALGORITHM COMPARISON: {M}×{N}×{K}")
    print(f"{'='*80}")

    print(f"\n{'Algorithm':<25s} {'Best GFLOP/s':>14s} {'Evals':>8s} {'Wall Time':>12s} {'Config':>20s}")
    print("-" * 80)

    algo_names = {
        'gradient_descent': 'Gradient Descent',
        'bayesian_optimization': 'Bayesian Optimization',
        'simulated_annealing': 'Simulated Annealing',
        'random_search': 'Random Search',
    }

    best_overall = max(r.best_gflops for r in results.values())

    for key in ['gradient_descent', 'bayesian_optimization', 'simulated_annealing', 'random_search']:
        result = results.get(key)
        if result is None:
            continue

        cfg = result.best_config
        config_str = f"MC={cfg.MC} KC={cfg.KC} NC={cfg.NC}"
        marker = " ★" if result.best_gflops == best_overall else ""

        print(f"{algo_names.get(key, key):<25s} "
              f"{result.best_gflops:>12.2f}{marker:>2s} "
              f"{result.total_evaluations:>8d} "
              f"{result.wall_time_s:>10.1f}s "
              f"{config_str:>20s}")

    print("-" * 80)
    print(f"★ = winner\n")


def save_results(results: dict, M: int, N: int, K: int, kernel: str, threads: int):
    """Save comparison results to files."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Summary JSON
    summary = {}
    for key, result in results.items():
        cfg = result.best_config
        summary[key] = {
            'best_gflops': result.best_gflops,
            'best_config': {'MC': cfg.MC, 'KC': cfg.KC, 'NC': cfg.NC},
            'total_evaluations': result.total_evaluations,
            'wall_time_s': result.wall_time_s,
        }

    filepath = os.path.join(RESULTS_DIR, f"algo_comparison_{M}x{N}x{K}_t{threads}.json")
    with open(filepath, 'w') as f:
        json.dump({
            'problem': {'M': M, 'N': N, 'K': K},
            'kernel': kernel,
            'threads': threads,
            'algorithms': summary,
        }, f, indent=2)
    print(f"Results saved to: {filepath}")

    # Convergence CSV (for plotting)
    conv_file = os.path.join(RESULTS_DIR, f"convergence_{M}x{N}x{K}_t{threads}.csv")
    with open(conv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        max_evals = max(len(r.history) for r in results.values())
        header = ['evaluation'] + list(results.keys())
        writer.writerow(header)

        curves = {k: compute_convergence(v.history) for k, v in results.items()}

        for i in range(max_evals):
            row = [i + 1]
            for key in results:
                curve = curves[key]
                row.append(f"{curve[i]:.4f}" if i < len(curve) else "")
            writer.writerow(row)
    print(f"Convergence data saved to: {conv_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare optimization algorithms for SGEMM block sizes"
    )
    parser.add_argument('--M', type=int, default=1024)
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--K', type=int, default=None)
    parser.add_argument('--kernel', default='6x16')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--budget', type=int, default=30,
                        help='Evaluation budget per algorithm')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    M, N, K = args.M, args.N or args.M, args.K or args.M

    print("=" * 80)
    print("SGEMM Autotuning Algorithm Comparison")
    print("=" * 80)
    print(f"Problem: {M} × {N} × {K}")
    print(f"Kernel: {args.kernel}, Threads: {args.threads}")
    print(f"Budget: {args.budget} evaluations per algorithm")
    print()

    results = run_comparison(M, N, K, args.kernel, args.threads,
                             args.budget, args.verbose)

    print_summary(results, M, N, K)
    save_results(results, M, N, K, args.kernel, args.threads)

    return 0


if __name__ == '__main__':
    sys.exit(main())
