#!/usr/bin/env python3
"""
random_search.py — Random Search baseline for SGEMM block sizes

Serves as a baseline to compare against smarter algorithms (gradient descent,
Bayesian, simulated annealing). Surprisingly competitive for high-dimensional
spaces, and if it outperforms an 'intelligent' algorithm, that algorithm has
a problem.

Usage:
    python3 random_search.py --M 1024 --threads 4
    python3 random_search.py --M 512 --iterations 50 -v
"""

import os
import sys
import json
import argparse
import time
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_runner import BenchmarkRunner, SGEMMConfig
from gradient_descent import OptimizationResult

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


class RandomSearchOptimizer:
    """
    Pure random sampling within the valid parameter space.

    Tests random (MC, KC, NC) configurations and returns the best found.
    """

    def __init__(self, runner: BenchmarkRunner, M: int, N: int = None, K: int = None,
                 kernel: str = "6x16", threads: int = 1, verbose: bool = False):
        self.runner = runner
        self.M = M
        self.N = N or M
        self.K = K or M
        self.kernel = kernel
        self.threads = threads
        self.verbose = verbose
        self.history = []

        kernel_dims = {"8x8": (8, 8), "6x16": (6, 16), "4x24": (4, 24), "6x16_asm": (6, 16)}
        self.MR, self.NR = kernel_dims.get(kernel, (6, 16))

    def _evaluate(self, MC: int, KC: int, NC: int) -> float:
        """Run real benchmark."""
        config = SGEMMConfig(
            M=self.M, N=self.N, K=self.K,
            kernel=self.kernel, threads=self.threads,
            MC=int(MC), KC=int(KC), NC=int(NC)
        )

        if not self.runner.is_valid_config(config):
            return 0.0

        result = self.runner.run(config)
        gflops = result.median_gflops if result.success else 0.0

        self.history.append({
            'MC': int(MC), 'KC': int(KC), 'NC': int(NC), 'gflops': gflops
        })

        if self.verbose:
            print(f"    MC={int(MC):4d} KC={int(KC):4d} NC={int(NC):5d} → {gflops:7.2f} GFLOP/s")

        return gflops

    def optimize(self, n_iterations: int = 50, seed: int = 42) -> OptimizationResult:
        """
        Run random search.

        Args:
            n_iterations: Number of random configurations to try
            seed: Random seed for reproducibility
        """
        start_time = time.time()
        self.history = []
        np.random.seed(seed)

        # Define valid values
        mc_values = [i * self.MR for i in range(1, 480 // self.MR + 1)]
        kc_values = list(range(64, 1537, 64))
        nc_values = [i * self.NR for i in range(1, 16384 // self.NR + 1)]

        best_mc, best_kc, best_nc = 120, 512, 4096
        best_score = 0.0

        # Always evaluate the known-good default first
        score = self._evaluate(120, 512, 4096)
        if score > best_score:
            best_score = score
            best_mc, best_kc, best_nc = 120, 512, 4096

        for i in range(n_iterations - 1):
            mc = np.random.choice(mc_values)
            kc = np.random.choice(kc_values)
            nc = np.random.choice(nc_values)

            score = self._evaluate(mc, kc, nc)

            if score > best_score:
                best_score = score
                best_mc, best_kc, best_nc = mc, kc, nc
                if self.verbose:
                    print(f"  *** New best at iter {i+1}: {best_score:.2f} GFLOP/s")

        wall_time = time.time() - start_time

        best_config = SGEMMConfig(
            M=self.M, N=self.N, K=self.K,
            kernel=self.kernel, threads=self.threads,
            MC=best_mc, KC=best_kc, NC=best_nc
        )

        return OptimizationResult(
            best_config=best_config,
            best_gflops=best_score,
            total_evaluations=len(self.history),
            history=self.history,
            algorithm="random_search",
            wall_time_s=wall_time
        )


def main():
    parser = argparse.ArgumentParser(
        description="Random search baseline for SGEMM block sizes (real benchmarks)"
    )
    parser.add_argument('--M', type=int, default=1024)
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--K', type=int, default=None)
    parser.add_argument('--kernel', default='6x16')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('-o', '--output', default=None)
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    M, N, K = args.M, args.N or args.M, args.K or args.M

    print("=" * 70)
    print("SGEMM Block Size Optimization — Random Search (Baseline)")
    print("=" * 70)
    print(f"Problem: {M} × {N} × {K}")
    print(f"Kernel: {args.kernel}, Threads: {args.threads}")
    print(f"Iterations: {args.iterations}")
    print()

    runner = BenchmarkRunner(verbose=args.verbose)
    optimizer = RandomSearchOptimizer(
        runner, M, N, K,
        kernel=args.kernel, threads=args.threads, verbose=args.verbose
    )

    result = optimizer.optimize(n_iterations=args.iterations)

    print(f"\n{'='*70}")
    print("RESULT")
    print(f"{'='*70}")
    cfg = result.best_config
    print(f"Best: MC={cfg.MC} KC={cfg.KC} NC={cfg.NC}")
    print(f"Performance: {result.best_gflops:.2f} GFLOP/s")
    print(f"Evaluations: {result.total_evaluations}")
    print(f"Wall time: {result.wall_time_s:.1f}s")

    output_file = args.output or f"rs_optimal_{M}x{N}x{K}.json"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, output_file)

    with open(filepath, 'w') as f:
        json.dump({
            'algorithm': 'random_search',
            'problem_size': {'M': M, 'N': N, 'K': K},
            'kernel': args.kernel, 'threads': args.threads,
            'optimal_config': {'MC': cfg.MC, 'KC': cfg.KC, 'NC': cfg.NC},
            'best_gflops': result.best_gflops,
            'total_evaluations': result.total_evaluations,
            'wall_time_s': result.wall_time_s,
        }, f, indent=2)
    print(f"\nSaved to: {filepath}")

    history_file = os.path.join(RESULTS_DIR, f"rs_history_{M}x{N}x{K}.csv")
    with open(history_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['MC', 'KC', 'NC', 'gflops'])
        writer.writeheader()
        writer.writerows(result.history)
    print(f"History saved to: {history_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
