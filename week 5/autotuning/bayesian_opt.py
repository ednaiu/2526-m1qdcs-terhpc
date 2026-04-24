#!/usr/bin/env python3
"""
bayesian_opt.py — Bayesian Optimization for SGEMM block sizes (real benchmarks)

Uses Gaussian Processes to model the performance surface and
Expected Improvement acquisition function to intelligently explore
the block size parameter space.

Advantages over gradient descent:
    - Global optimization (doesn't get stuck at local optima)
    - Model-based: builds a surrogate surface and reasons about uncertainty
    - Sample-efficient: typically finds good configs in fewer evaluations

Dependencies:
    - numpy (required)
    - scipy (optional, for better GP fitting)

Usage:
    python3 bayesian_opt.py --M 1024 --threads 4
    python3 bayesian_opt.py --M 512 --iterations 30 --init-samples 5 -v
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
from benchmark_runner import BenchmarkRunner, SGEMMConfig, BenchmarkResult

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

try:
    from scipy.stats import norm
    from scipy.optimize import minimize as scipy_minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class GaussianProcess:
    """Simple Gaussian Process regression for Bayesian Optimization."""

    def __init__(self, length_scale: float = 100.0, noise: float = 0.1):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha_ = None

    def _rbf_kernel(self, X1, X2):
        """Squared exponential (RBF) kernel."""
        sq_dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        return np.exp(-sq_dists / (2.0 * self.length_scale ** 2))

    def fit(self, X, y):
        """Fit GP to training data."""
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=float)

        K = self._rbf_kernel(self.X_train, self.X_train)
        K += np.eye(len(self.X_train)) * (self.noise ** 2 + 1e-8)

        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            self.K_inv = np.linalg.inv(K + np.eye(len(K)) * 1e-4)

        self.alpha_ = self.K_inv @ self.y_train

    def predict(self, X_test):
        """Predict mean and std at test points."""
        X_test = np.array(X_test, dtype=float)

        if self.X_train is None or len(self.X_train) == 0:
            return np.zeros(len(X_test)), np.ones(len(X_test))

        K_star = self._rbf_kernel(self.X_train, X_test)
        mean = K_star.T @ self.alpha_

        K_star_star = self._rbf_kernel(X_test, X_test)
        var = np.diag(K_star_star) - np.sum(K_star * (self.K_inv @ K_star), axis=0)
        var = np.maximum(var, 1e-8)
        std = np.sqrt(var)

        return mean, std

    def expected_improvement(self, X_test, y_best, xi=0.01):
        """Expected Improvement acquisition function."""
        mean, std = self.predict(X_test)

        with np.errstate(divide='ignore', invalid='ignore'):
            Z = (mean - y_best - xi) / std
            if HAS_SCIPY:
                ei = (mean - y_best - xi) * norm.cdf(Z) + std * norm.pdf(Z)
            else:
                # Approximate normal CDF/PDF
                ei = (mean - y_best - xi) * (0.5 + 0.5 * np.tanh(Z * 0.7978)) + \
                     std * (0.3989 * np.exp(-0.5 * Z ** 2))

        ei[std < 1e-9] = 0.0
        return ei


class BayesianOptimizer:
    """
    Bayesian Optimization for SGEMM block sizes.

    Uses real benchmark execution for evaluation and a Gaussian Process
    surrogate model with Expected Improvement acquisition.
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

        # Define parameter space
        self.mc_values = np.array([i * self.MR for i in range(1, 1024 // self.MR + 1)])
        self.kc_values = np.arange(64, 2049, 64)
        self.nc_values = np.array([i * self.NR for i in range(1, 32768 // self.NR + 1)])

        # Filter to valid configurations
        self.mc_values = self.mc_values[(self.mc_values >= self.MR) & (self.mc_values <= 480)]
        self.kc_values = self.kc_values[(self.kc_values >= 64) & (self.kc_values <= 1536)]
        self.nc_values = self.nc_values[(self.nc_values >= self.NR) & (self.nc_values <= 16384)]

        # New parallelism dimensions
        self.parallel_modes = [0, 1, 2]  # 2D, 3D, DYNAMIC
        self.r_values = [1, 2, 4, 8]

    def _evaluate(self, MC: int, KC: int, NC: int, parallel: int, r: int) -> float:
        """Run real benchmark."""
        parallel_str = ["2D", "3D", "DYNAMIC"][int(parallel)]
        config = SGEMMConfig(
            M=self.M, N=self.N, K=self.K,
            kernel=self.kernel, threads=self.threads,
            MC=int(MC), KC=int(KC), NC=int(NC),
            parallel=parallel_str, r_tasks=int(r)
        )

        if not self.runner.is_valid_config(config):
            return 0.0

        result = self.runner.run(config)
        gflops = result.median_gflops if result.success else 0.0

        self.history.append({
            'MC': int(MC), 'KC': int(KC), 'NC': int(NC),
            'parallel': parallel_str, 'r': int(r),
            'gflops': gflops,
        })

        if self.verbose:
            para_info = f"{parallel_str}(r={int(r)})"
            print(f"    MC={int(MC):4d} KC={int(KC):4d} NC={int(NC):5d}  {para_info:10s} → {gflops:7.2f} GFLOP/s")

        return gflops

    def _get_initial_samples(self, n_init: int):
        """Generate diverse initial samples."""
        samples = [
            (120, 512, 4096, 2, 1),   # DYNAMIC, r=1
            (120, 512, 4096, 0, 1),   # 2D, r=1
            (96, 256, 2048, 1, 2),    # 3D, r=2
            (168, 384, 4096, 2, 2),   # DYNAMIC, r=2
            (120, 768, 4096, 0, 1),   # 2D
        ]

        # Add random samples if needed
        np.random.seed(42)
        while len(samples) < n_init:
            mc = np.random.choice(self.mc_values)
            kc = np.random.choice(self.kc_values)
            nc = np.random.choice(self.nc_values)
            p  = np.random.choice(self.parallel_modes)
            r  = np.random.choice(self.r_values)
            samples.append((int(mc), int(kc), int(nc), int(p), int(r)))

        return samples[:n_init]

    def _generate_candidates(self, n_candidates: int = 200):
        """Generate random candidate configurations for EI evaluation."""
        candidates = []
        for _ in range(n_candidates):
            mc = np.random.choice(self.mc_values)
            kc = np.random.choice(self.kc_values)
            nc = np.random.choice(self.nc_values)
            p  = np.random.choice(self.parallel_modes)
            r  = np.random.choice(self.r_values)
            parallel_str = ["2D", "3D", "DYNAMIC"][int(p)]
            config = SGEMMConfig(
                M=self.M, N=self.N, K=self.K,
                kernel=self.kernel, threads=self.threads,
                MC=int(mc), KC=int(kc), NC=int(nc),
                parallel=parallel_str, r_tasks=int(r)
            )
            if self.runner.is_valid_config(config):
                candidates.append([mc, kc, nc, p, r])
        return np.array(candidates)

    def optimize(self, n_iterations: int = 25, n_init: int = 5):
        """
        Run Bayesian optimization.

        Args:
            n_iterations: Total iterations (including initial sampling)
            n_init: Number of initial random samples
        """
        start_time = time.time()
        self.history = []

        # Phase 1: Initial sampling
        if self.verbose:
            print(f"Phase 1: Initial sampling ({n_init} points)...")

        init_samples = self._get_initial_samples(n_init)
        X_train = []
        y_train = []

        for mc, kc, nc, p, r in init_samples:
            score = self._evaluate(mc, kc, nc, p, r)
            X_train.append([mc, kc, nc, p, r])
            y_train.append(score)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Phase 2: Bayesian optimization
        if self.verbose:
            print(f"\nPhase 2: Bayesian optimization ({n_iterations - n_init} iterations)...")

        gp = GaussianProcess(length_scale=150.0, noise=0.5)

        for iteration in range(n_iterations - n_init):
            # Fit GP
            gp.fit(X_train, y_train)
            y_best = np.max(y_train)

            # Generate candidates and compute EI
            candidates = self._generate_candidates(300)
            if len(candidates) == 0:
                break

            ei = gp.expected_improvement(candidates, y_best, xi=0.1)

            # Select best candidate
            best_idx = np.argmax(ei)
            next_point = candidates[best_idx]

            # Evaluate
            mc, kc, nc, p, r = [int(next_point[i]) for i in range(5)]
            score = self._evaluate(mc, kc, nc, p, r)

            if self.verbose:
                print(f"  BO iter {iteration+1:2d}: EI={ei[best_idx]:.4f}")

            X_train = np.vstack([X_train, next_point])
            y_train = np.append(y_train, score)

        # Find best
        best_idx = np.argmax(y_train)
        best_mc, best_kc, best_nc, best_p, best_r = X_train[best_idx]
        best_gflops = y_train[best_idx]
        best_para = ["2D", "3D", "DYNAMIC"][int(best_p)]

        wall_time = time.time() - start_time

        best_config = SGEMMConfig(
            M=self.M, N=self.N, K=self.K,
            kernel=self.kernel, threads=self.threads,
            MC=int(best_mc), KC=int(best_kc), NC=int(best_nc),
            parallel=best_para, r_tasks=int(best_r)
        )

        from gradient_descent import OptimizationResult
        return OptimizationResult(
            best_config=best_config,
            best_gflops=best_gflops,
            total_evaluations=len(self.history),
            history=self.history,
            algorithm="bayesian_optimization",
            wall_time_s=wall_time
        )


def main():
    parser = argparse.ArgumentParser(
        description="Optimize SGEMM block sizes using Bayesian Optimization (real benchmarks)"
    )
    parser.add_argument('--M', type=int, default=1024)
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--K', type=int, default=None)
    parser.add_argument('--kernel', default='6x16')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=25)
    parser.add_argument('--init-samples', type=int, default=5)
    parser.add_argument('-o', '--output', default=None)
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    M = args.M
    N = args.N or M
    K = args.K or M

    print("=" * 70)
    print("SGEMM Block Size Optimization — Bayesian Optimization (Real Benchmarks)")
    print("=" * 70)
    print(f"Problem: {M} × {N} × {K}")
    print(f"Kernel: {args.kernel}, Threads: {args.threads}")
    print(f"Iterations: {args.iterations} ({args.init_samples} init + {args.iterations - args.init_samples} BO)")
    print()

    runner = BenchmarkRunner(verbose=args.verbose)
    optimizer = BayesianOptimizer(
        runner, M, N, K,
        kernel=args.kernel, threads=args.threads,
        verbose=args.verbose
    )

    result = optimizer.optimize(
        n_iterations=args.iterations,
        n_init=args.init_samples
    )

    print(f"\n{'='*70}")
    print("RESULT")
    print(f"{'='*70}")
    cfg = result.best_config
    print(f"Best: MC={cfg.MC} KC={cfg.KC} NC={cfg.NC}")
    print(f"      Parallelism: {cfg.parallel} (r={cfg.r_tasks})")
    print(f"Performance: {result.best_gflops:.2f} GFLOP/s")
    print(f"Evaluations: {result.total_evaluations}")
    print(f"Wall time: {result.wall_time_s:.1f}s")

    # Save result
    output_file = args.output or f"bo_optimal_{M}x{N}x{K}.json"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, output_file)

    with open(filepath, 'w') as f:
        json.dump({
            'algorithm': 'bayesian_optimization',
            'problem_size': {'M': M, 'N': N, 'K': K},
            'kernel': args.kernel,
            'threads': args.threads,
            'optimal_config': {
                'MC': cfg.MC, 'KC': cfg.KC, 'NC': cfg.NC,
                'parallel': cfg.parallel, 'r_tasks': cfg.r_tasks
            },
            'best_gflops': result.best_gflops,
            'total_evaluations': result.total_evaluations,
            'wall_time_s': result.wall_time_s,
        }, f, indent=2)
    print(f"\nSaved to: {filepath}")

    # Save history
    history_file = os.path.join(RESULTS_DIR, f"bo_history_{M}x{N}x{K}.csv")
    with open(history_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['MC', 'KC', 'NC', 'parallel', 'r', 'gflops'])
        writer.writeheader()
        writer.writerows(result.history)
    print(f"History saved to: {history_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
