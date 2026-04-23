#!/usr/bin/env python3
"""
simulated_annealing.py — Simulated Annealing optimization for SGEMM block sizes

Alternative to gradient descent and Bayesian optimization.
Uses temperature-based acceptance of worse solutions to escape local minima.

Key difference from gradient descent:
    - Can accept worse solutions with probability exp(-ΔE / T)
    - As temperature decreases, becomes more greedy → converges to optimum
    - Global exploration at high T, local refinement at low T

Usage:
    python3 simulated_annealing.py --M 1024 --threads 4
    python3 simulated_annealing.py --M 512 --iterations 50 -v
"""

import os
import sys
import json
import argparse
import time
import csv
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_runner import BenchmarkRunner, SGEMMConfig
from gradient_descent import OptimizationResult

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


class SimulatedAnnealingOptimizer:
    """
    Simulated Annealing for SGEMM block size optimization.

    Uses real benchmark execution and a cooling schedule to balance
    exploration vs exploitation.
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

    def _neighbor(self, MC: int, KC: int, NC: int, temperature: float) -> tuple:
        """
        Generate a neighbor configuration.

        Step size scales with temperature:
        - High T → large perturbations (exploration)
        - Low T → small perturbations (refinement)
        """
        # Scale perturbation with temperature
        scale = max(1, int(temperature / 10))

        mc_delta = np.random.choice([-2, -1, 0, 1, 2]) * self.MR * scale
        kc_delta = np.random.choice([-2, -1, 0, 1, 2]) * 64 * scale
        nc_delta = np.random.choice([-4, -2, -1, 0, 1, 2, 4]) * self.NR * scale

        new_mc = max(self.MR, MC + mc_delta)
        new_kc = max(64, KC + kc_delta)
        new_nc = max(self.NR, NC + nc_delta)

        # Round to valid multiples
        new_mc = self.MR * round(new_mc / self.MR)
        new_kc = 64 * round(new_kc / 64)
        new_nc = self.NR * round(new_nc / self.NR)

        # Clamp to valid range
        new_mc = max(self.MR, min(480, new_mc))
        new_kc = max(64, min(1536, new_kc))
        new_nc = max(self.NR, min(16384, new_nc))

        return int(new_mc), int(new_kc), int(new_nc)

    def optimize(self, n_iterations: int = 50,
                 initial_temp: float = 50.0,
                 cooling_rate: float = 0.92,
                 initial_MC: int = 120, initial_KC: int = 512,
                 initial_NC: int = 4096) -> OptimizationResult:
        """
        Run simulated annealing optimization.

        Args:
            n_iterations: Total number of iterations
            initial_temp: Starting temperature
            cooling_rate: Geometric cooling factor (T *= cooling_rate each iteration)
            initial_MC, initial_KC, initial_NC: Starting point
        """
        start_time = time.time()
        self.history = []

        MC, KC, NC = initial_MC, initial_KC, initial_NC
        current_score = self._evaluate(MC, KC, NC)

        best_mc, best_kc, best_nc = MC, KC, NC
        best_score = current_score

        temperature = initial_temp

        for iteration in range(n_iterations):
            # Generate neighbor
            new_mc, new_kc, new_nc = self._neighbor(MC, KC, NC, temperature)
            new_score = self._evaluate(new_mc, new_kc, new_nc)

            if new_score <= 0:
                # Invalid config, skip
                temperature *= cooling_rate
                continue

            # Accept or reject
            delta = new_score - current_score  # Positive = improvement (we maximize)

            if delta > 0:
                # Always accept improvements
                accept = True
            else:
                # Accept worse solutions with probability exp(delta / T)
                if temperature > 0.01:
                    prob = np.exp(delta / temperature)
                    accept = np.random.random() < prob
                else:
                    accept = False

            if accept:
                MC, KC, NC = new_mc, new_kc, new_nc
                current_score = new_score

                if current_score > best_score:
                    best_mc, best_kc, best_nc = MC, KC, NC
                    best_score = current_score

            if self.verbose:
                print(f"  Iter {iteration+1:3d} T={temperature:6.2f}: "
                      f"MC={MC:4d} KC={KC:4d} NC={NC:5d} score={current_score:7.2f} "
                      f"best={best_score:7.2f} {'✓' if accept else '✗'}")

            temperature *= cooling_rate

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
            algorithm="simulated_annealing",
            wall_time_s=wall_time
        )


def main():
    parser = argparse.ArgumentParser(
        description="Optimize SGEMM block sizes using Simulated Annealing (real benchmarks)"
    )
    parser.add_argument('--M', type=int, default=1024)
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--K', type=int, default=None)
    parser.add_argument('--kernel', default='6x16')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--initial-temp', type=float, default=50.0)
    parser.add_argument('--cooling-rate', type=float, default=0.92)
    parser.add_argument('-o', '--output', default=None)
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    M, N, K = args.M, args.N or args.M, args.K or args.M

    print("=" * 70)
    print("SGEMM Block Size Optimization — Simulated Annealing (Real Benchmarks)")
    print("=" * 70)
    print(f"Problem: {M} × {N} × {K}")
    print(f"Kernel: {args.kernel}, Threads: {args.threads}")
    print(f"Iterations: {args.iterations}, T₀={args.initial_temp}, α={args.cooling_rate}")
    print()

    runner = BenchmarkRunner(verbose=args.verbose)
    optimizer = SimulatedAnnealingOptimizer(
        runner, M, N, K,
        kernel=args.kernel, threads=args.threads, verbose=args.verbose
    )

    result = optimizer.optimize(
        n_iterations=args.iterations,
        initial_temp=args.initial_temp,
        cooling_rate=args.cooling_rate
    )

    print(f"\n{'='*70}")
    print("RESULT")
    print(f"{'='*70}")
    cfg = result.best_config
    print(f"Best: MC={cfg.MC} KC={cfg.KC} NC={cfg.NC}")
    print(f"Performance: {result.best_gflops:.2f} GFLOP/s")
    print(f"Evaluations: {result.total_evaluations}")
    print(f"Wall time: {result.wall_time_s:.1f}s")

    output_file = args.output or f"sa_optimal_{M}x{N}x{K}.json"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, output_file)

    with open(filepath, 'w') as f:
        json.dump({
            'algorithm': 'simulated_annealing',
            'problem_size': {'M': M, 'N': N, 'K': K},
            'kernel': args.kernel, 'threads': args.threads,
            'optimal_config': {'MC': cfg.MC, 'KC': cfg.KC, 'NC': cfg.NC},
            'best_gflops': result.best_gflops,
            'total_evaluations': result.total_evaluations,
            'wall_time_s': result.wall_time_s,
        }, f, indent=2)
    print(f"\nSaved to: {filepath}")

    history_file = os.path.join(RESULTS_DIR, f"sa_history_{M}x{N}x{K}.csv")
    with open(history_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['MC', 'KC', 'NC', 'gflops'])
        writer.writeheader()
        writer.writerows(result.history)
    print(f"History saved to: {history_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
