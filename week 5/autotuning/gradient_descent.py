#!/usr/bin/env python3
"""
gradient_descent.py — Gradient descent optimization for SGEMM block sizes

Uses REAL benchmark execution (not a theoretical model) to optimize
cache blocking parameters (MC, KC, NC) for different problem sizes.

Algorithm:
    - Finite-difference gradient estimation (coordinate descent)
    - Line search along gradient direction
    - Random restarts to escape local minima
    - Constraint checking (cache hierarchy, MR/NR alignment)

Limitations:
    - Gradient descent is a local optimizer → can get stuck at local optima
    - Starting point matters → random restarts help but don't guarantee global optimum
    - Uses discrete grid (multiples of MR/NR) → step sizes are coarse

Usage:
    python3 gradient_descent.py --M 1024 --threads 4
    python3 gradient_descent.py --M 512 --restarts 3 --max-iter 15 -v
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
from dataclasses import dataclass, asdict

# Import the generic benchmark runner
sys.path.insert(0, os.path.dirname(__file__))
from benchmark_runner import BenchmarkRunner, SGEMMConfig, BenchmarkResult

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


@dataclass
class OptimizationResult:
    """Result from optimization run."""
    best_config: SGEMMConfig
    best_gflops: float
    total_evaluations: int
    history: list  # list of (config_dict, gflops) tuples
    algorithm: str
    wall_time_s: float


class GradientDescentOptimizer:
    """
    Optimize SGEMM block sizes using coordinate gradient descent.

    Uses real benchmark execution — each function evaluation runs the
    actual SGEMM kernel and measures execution time.
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

        # Get MR/NR for this kernel
        kernel_dims = {"8x8": (8, 8), "6x16": (6, 16), "4x24": (4, 24), "6x16_asm": (6, 16)}
        self.MR, self.NR = kernel_dims.get(kernel, (6, 16))

    def _evaluate(self, MC: int, KC: int, NC: int) -> float:
        """Run actual benchmark and return GFLOP/s (or 0 on failure)."""
        config = SGEMMConfig(
            M=self.M, N=self.N, K=self.K,
            kernel=self.kernel, threads=self.threads,
            MC=MC, KC=KC, NC=NC
        )

        if not self.runner.is_valid_config(config):
            return 0.0

        result = self.runner.run(config)
        gflops = result.median_gflops if result.success else 0.0

        self.history.append({
            'MC': MC, 'KC': KC, 'NC': NC,
            'gflops': gflops,
            'runs': result.runs if result.success else 0,
            'stddev_pct': result.stddev_pct if result.success else 0,
        })

        if self.verbose:
            print(f"    MC={MC:4d} KC={KC:4d} NC={NC:5d} → {gflops:7.2f} GFLOP/s")

        return gflops

    def _round_mc(self, val: int) -> int:
        """Round to nearest multiple of MR."""
        return max(self.MR, self.MR * round(val / self.MR))

    def _round_nc(self, val: int) -> int:
        """Round to nearest multiple of NR."""
        return max(self.NR, self.NR * round(val / self.NR))

    def _round_kc(self, val: int) -> int:
        """Round KC to nearest multiple of 64."""
        return max(64, 64 * round(val / 64))

    def coordinate_descent_step(self, MC: int, KC: int, NC: int) -> tuple:
        """
        One step of coordinate descent: try perturbing each parameter independently.

        Returns: (new_MC, new_KC, new_NC, new_score, improved)
        """
        best_score = self._evaluate(MC, KC, NC)
        best_params = (MC, KC, NC)
        improved = False

        # MC perturbations
        for delta in [self.MR, -self.MR, 2*self.MR, -2*self.MR]:
            new_mc = self._round_mc(MC + delta)
            if new_mc != MC:
                score = self._evaluate(new_mc, KC, NC)
                if score > best_score:
                    best_score = score
                    best_params = (new_mc, KC, NC)
                    improved = True

        # KC perturbations
        for delta in [64, -64, 128, -128, 256, -256]:
            new_kc = self._round_kc(KC + delta)
            if new_kc != KC:
                score = self._evaluate(MC, new_kc, NC)
                if score > best_score:
                    best_score = score
                    best_params = (MC, new_kc, NC)
                    improved = True

        # NC perturbations
        for delta in [self.NR*4, -self.NR*4, self.NR*16, -self.NR*16, self.NR*64, -self.NR*64]:
            new_nc = self._round_nc(NC + delta)
            if new_nc != NC:
                score = self._evaluate(MC, KC, new_nc)
                if score > best_score:
                    best_score = score
                    best_params = (MC, KC, new_nc)
                    improved = True

        return best_params[0], best_params[1], best_params[2], best_score, improved

    def optimize(self, initial_MC: int = 120, initial_KC: int = 512,
                 initial_NC: int = 4096, max_iterations: int = 15) -> OptimizationResult:
        """
        Run gradient descent optimization from a single starting point.

        Args:
            initial_MC, initial_KC, initial_NC: Starting block sizes
            max_iterations: Max optimization steps

        Returns:
            OptimizationResult with best configuration found
        """
        start_time = time.time()
        MC, KC, NC = initial_MC, initial_KC, initial_NC
        best_gflops = 0.0

        if self.verbose:
            print(f"  Start: MC={MC} KC={KC} NC={NC}")

        for iteration in range(max_iterations):
            MC_new, KC_new, NC_new, score, improved = \
                self.coordinate_descent_step(MC, KC, NC)

            if self.verbose:
                print(f"  Iter {iteration+1:2d}: MC={MC_new:4d} KC={KC_new:4d} NC={NC_new:5d}"
                      f" → {score:7.2f} GFLOP/s {'↑' if improved else '='}")

            if not improved:
                break

            MC, KC, NC = MC_new, KC_new, NC_new
            best_gflops = score

        wall_time = time.time() - start_time

        best_config = SGEMMConfig(
            M=self.M, N=self.N, K=self.K,
            kernel=self.kernel, threads=self.threads,
            MC=MC, KC=KC, NC=NC
        )

        return OptimizationResult(
            best_config=best_config,
            best_gflops=best_gflops,
            total_evaluations=len(self.history),
            history=self.history.copy(),
            algorithm="gradient_descent",
            wall_time_s=wall_time
        )

    def optimize_with_restarts(self, n_restarts: int = 3,
                                max_iterations: int = 15) -> OptimizationResult:
        """
        Run gradient descent from multiple random starting points.

        Best overall result is returned (helps escape local minima).
        """
        # Starting points: known good defaults + random
        starting_points = [
            (120, 512, 4096),   # Known good (Haswell default)
            (96, 256, 2048),    # Smaller blocks
            (120, 768, 4096),   # Larger KC
        ]

        # Add random starting points
        np.random.seed(42)
        for _ in range(max(0, n_restarts - len(starting_points))):
            mc = self._round_mc(np.random.randint(6, 30) * self.MR)
            kc = self._round_kc(np.random.randint(1, 16) * 64)
            nc = self._round_nc(np.random.randint(1, 512) * self.NR)
            starting_points.append((mc, kc, nc))

        best_result = None

        for i, (mc, kc, nc) in enumerate(starting_points[:n_restarts]):
            if self.verbose:
                print(f"\n--- Restart {i+1}/{n_restarts}: MC={mc} KC={kc} NC={nc} ---")

            self.history = []
            result = self.optimize(mc, kc, nc, max_iterations)

            if best_result is None or result.best_gflops > best_result.best_gflops:
                best_result = result

        if self.verbose:
            print(f"\nBest across all restarts: {best_result.best_gflops:.2f} GFLOP/s")
            cfg = best_result.best_config
            print(f"  MC={cfg.MC} KC={cfg.KC} NC={cfg.NC}")

        return best_result


def main():
    parser = argparse.ArgumentParser(
        description="Optimize SGEMM block sizes using gradient descent (real benchmarks)"
    )
    parser.add_argument('--M', type=int, default=1024)
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--K', type=int, default=None)
    parser.add_argument('--kernel', default='6x16')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--restarts', type=int, default=3,
                        help='Number of random restarts')
    parser.add_argument('--max-iter', type=int, default=15,
                        help='Max iterations per restart')
    parser.add_argument('-o', '--output', default=None)
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    M = args.M
    N = args.N or M
    K = args.K or M

    print("=" * 70)
    print("SGEMM Block Size Optimization — Gradient Descent (Real Benchmarks)")
    print("=" * 70)
    print(f"Problem: {M} × {N} × {K}")
    print(f"Kernel: {args.kernel}, Threads: {args.threads}")
    print(f"Restarts: {args.restarts}, Max iterations: {args.max_iter}")
    print()

    runner = BenchmarkRunner(verbose=args.verbose)
    optimizer = GradientDescentOptimizer(
        runner, M, N, K,
        kernel=args.kernel, threads=args.threads,
        verbose=args.verbose
    )

    result = optimizer.optimize_with_restarts(
        n_restarts=args.restarts,
        max_iterations=args.max_iter
    )

    print(f"\n{'='*70}")
    print("RESULT")
    print(f"{'='*70}")
    cfg = result.best_config
    print(f"Best: MC={cfg.MC} KC={cfg.KC} NC={cfg.NC}")
    print(f"Performance: {result.best_gflops:.2f} GFLOP/s")
    print(f"Evaluations: {result.total_evaluations}")
    print(f"Wall time: {result.wall_time_s:.1f}s")

    # Save result
    output_file = args.output or f"gd_optimal_{M}x{N}x{K}.json"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, output_file)

    with open(filepath, 'w') as f:
        json.dump({
            'algorithm': 'gradient_descent',
            'problem_size': {'M': M, 'N': N, 'K': K},
            'kernel': args.kernel,
            'threads': args.threads,
            'optimal_config': {'MC': cfg.MC, 'KC': cfg.KC, 'NC': cfg.NC},
            'best_gflops': result.best_gflops,
            'total_evaluations': result.total_evaluations,
            'wall_time_s': result.wall_time_s,
        }, f, indent=2)
    print(f"\nSaved to: {filepath}")

    # Save full history to CSV
    history_file = os.path.join(RESULTS_DIR, f"gd_history_{M}x{N}x{K}.csv")
    with open(history_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['MC', 'KC', 'NC', 'gflops', 'runs', 'stddev_pct'])
        writer.writeheader()
        writer.writerows(result.history)
    print(f"History saved to: {history_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
