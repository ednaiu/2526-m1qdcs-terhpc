#!/usr/bin/env python3
"""
benchmark_runner.py — Generic benchmark runner for SGEMM (and future BLAS kernels)

Provides a clean Python interface to the C benchmark binary.
All autotuning algorithms should use this module instead of calling
the binary directly.

Features:
    - Adaptive N: start with n=3, increase until stddev < 2%
    - Uses MEDIAN for central tendency
    - CPU affinity pinning via taskset
    - Timeout handling
    - JSON parsing from bench_sgemm single mode
    - Generic enough to reuse for BLAS1, BLAS2, etc.

Usage as module:
    from benchmark_runner import BenchmarkRunner, SGEMMConfig

    runner = BenchmarkRunner()
    config = SGEMMConfig(M=1024, N=1024, K=1024, kernel="6x16", threads=4)
    result = runner.run(config)
    print(f"Median: {result.median_gflops:.2f} GFLOP/s")

Usage as script:
    python3 benchmark_runner.py --M 1024 --kernel 6x16 --threads 4
"""

import subprocess
import json
import os
import sys
import time
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path


# Default path to benchmark binary (relative to week 2 directory)
DEFAULT_BENCH_BIN = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", "bench_sgemm")
if not os.path.isabs(DEFAULT_BENCH_BIN):
    DEFAULT_BENCH_BIN = os.path.abspath(DEFAULT_BENCH_BIN)


@dataclass
class SGEMMConfig:
    """Configuration for a single SGEMM benchmark run."""
    M: int = 1024
    N: Optional[int] = None       # Defaults to M
    K: Optional[int] = None       # Defaults to M
    kernel: str = "6x16"          # 8x8, 6x16, 4x24, 6x16_asm
    parallel: str = "2D"          # 2D, 3D
    threads: int = 1
    MC: int = 120
    KC: int = 512
    NC: int = 4096
    r_tasks: int = 1

    def __post_init__(self):
        if self.N is None:
            self.N = self.M
        if self.K is None:
            self.K = self.M

    def to_cli_args(self) -> List[str]:
        """Convert config to command-line arguments for bench_sgemm."""
        args = [
            f"--M", str(self.M),
            f"--N", str(self.N),
            f"--K", str(self.K),
            f"--kernel", self.kernel,
            f"--parallel", self.parallel,
            f"--threads", str(self.threads),
            f"--MC", str(self.MC),
            f"--KC", str(self.KC),
            f"--NC", str(self.NC),
            f"--r-tasks", str(self.r_tasks),
        ]
        return args

    def to_param_vector(self) -> List[int]:
        """Return tunable parameters as a list [MC, KC, NC]."""
        return [self.MC, self.KC, self.NC]

    @classmethod
    def from_param_vector(cls, params: List[int], **kwargs) -> 'SGEMMConfig':
        """Create config from parameter vector [MC, KC, NC] + other kwargs."""
        return cls(MC=int(params[0]), KC=int(params[1]), NC=int(params[2]), **kwargs)

    def param_key(self) -> str:
        """Unique key for this config (for caching)."""
        return f"{self.M}x{self.N}x{self.K}_{self.kernel}_{self.parallel}_t{self.threads}_MC{self.MC}_KC{self.KC}_NC{self.NC}_r{self.r_tasks}"


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    config: SGEMMConfig
    runs: int = 0
    median_s: float = 0.0
    mean_s: float = 0.0
    min_s: float = 0.0
    max_s: float = 0.0
    stddev_s: float = 0.0
    stddev_pct: float = 0.0
    median_gflops: float = 0.0
    mean_gflops: float = 0.0
    min_gflops: float = 0.0
    max_gflops: float = 0.0
    all_times: List[float] = field(default_factory=list)
    success: bool = True
    error: str = ""

    @classmethod
    def from_json(cls, data: Dict[str, Any], config: SGEMMConfig) -> 'BenchmarkResult':
        """Parse from bench_sgemm JSON output."""
        return cls(
            config=config,
            runs=data.get("runs", 0),
            median_s=data.get("median_s", 0.0),
            mean_s=data.get("mean_s", 0.0),
            min_s=data.get("min_s", 0.0),
            max_s=data.get("max_s", 0.0),
            stddev_s=data.get("stddev_s", 0.0),
            stddev_pct=data.get("stddev_pct", 0.0),
            median_gflops=data.get("gflops_median", 0.0),
            mean_gflops=data.get("gflops_mean", 0.0),
            min_gflops=data.get("gflops_min", 0.0),
            max_gflops=data.get("gflops_max", 0.0),
            all_times=data.get("all_times", []),
            success=True
        )

    @classmethod
    def failure(cls, config: SGEMMConfig, error: str) -> 'BenchmarkResult':
        """Create a failure result."""
        return cls(config=config, success=False, error=error)


class BenchmarkRunner:
    """
    Generic benchmark runner.

    Wraps a C benchmark binary and provides a clean Python interface.
    Can be extended for other kernels (BLAS1, BLAS2, etc.) by subclassing
    and overriding the binary path and config class.
    """

    def __init__(self, bench_bin: str = None, timeout: int = 120,
                 min_runs: int = 3, max_runs: int = 15,
                 target_stddev: float = 2.0,
                 use_taskset: bool = False,
                 verbose: bool = False):
        """
        Args:
            bench_bin: Path to benchmark binary
            timeout: Max seconds per benchmark invocation
            min_runs: Minimum timing samples
            max_runs: Maximum timing samples
            target_stddev: Target relative stddev (percent)
            use_taskset: Pin threads to first N CPUs
            verbose: Print progress
        """
        self.bench_bin = bench_bin or DEFAULT_BENCH_BIN
        self.timeout = timeout
        self.min_runs = min_runs
        self.max_runs = max_runs
        self.target_stddev = target_stddev
        self.use_taskset = use_taskset
        self.verbose = verbose
        self._cache: Dict[str, BenchmarkResult] = {}
        self.eval_count = 0

    def _check_binary(self):
        """Check that the benchmark binary exists."""
        if not os.path.exists(self.bench_bin):
            raise FileNotFoundError(
                f"Benchmark binary not found: {self.bench_bin}\n"
                f"Run 'make bench' in the week 2 directory first."
            )

    def run(self, config: SGEMMConfig, use_cache: bool = True) -> BenchmarkResult:
        """
        Run a single benchmark with the given configuration.

        Args:
            config: SGEMM configuration
            use_cache: If True, return cached result for identical configs

        Returns:
            BenchmarkResult with timing data
        """
        key = config.param_key()
        if use_cache and key in self._cache:
            if self.verbose:
                print(f"  [CACHED] {key}")
            return self._cache[key]

        self._check_binary()

        # Build command
        cmd = [self.bench_bin]
        cmd.extend(config.to_cli_args())
        cmd.extend([
            "--min-runs", str(self.min_runs),
            "--max-runs", str(self.max_runs),
            "--target-stddev", str(self.target_stddev),
        ])

        if self.use_taskset and config.threads > 0:
            cpu_list = ",".join(str(i) for i in range(config.threads))
            cmd = ["taskset", "-c", cpu_list] + cmd

        if self.verbose:
            print(f"  Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "OMP_NUM_THREADS": str(config.threads)}
            )

            if result.returncode != 0:
                err = result.stderr.strip() if result.stderr else "unknown error"
                if self.verbose:
                    print(f"  ERROR: {err}")
                return BenchmarkResult.failure(config, err)

            # Parse JSON output
            output = result.stdout.strip()
            if not output:
                return BenchmarkResult.failure(config, "empty output")

            data = json.loads(output)
            bench_result = BenchmarkResult.from_json(data, config)
            self.eval_count += 1

            if self.verbose:
                print(f"  → {bench_result.median_gflops:.2f} GFLOP/s "
                      f"(runs={bench_result.runs}, stddev={bench_result.stddev_pct:.1f}%)")

            if use_cache:
                self._cache[key] = bench_result
            return bench_result

        except subprocess.TimeoutExpired:
            return BenchmarkResult.failure(config, f"timeout after {self.timeout}s")
        except json.JSONDecodeError as e:
            return BenchmarkResult.failure(config, f"JSON parse error: {e}")
        except Exception as e:
            return BenchmarkResult.failure(config, str(e))

    def run_batch(self, configs: List[SGEMMConfig]) -> List[BenchmarkResult]:
        """Run multiple benchmarks sequentially."""
        results = []
        for i, cfg in enumerate(configs):
            if self.verbose:
                print(f"[{i+1}/{len(configs)}] {cfg.M}×{cfg.N}×{cfg.K} "
                      f"kernel={cfg.kernel} threads={cfg.threads} "
                      f"MC={cfg.MC} KC={cfg.KC} NC={cfg.NC}")
            results.append(self.run(cfg))
        return results

    def clear_cache(self):
        """Clear the results cache."""
        self._cache.clear()

    def is_valid_config(self, config: SGEMMConfig) -> bool:
        """Check if block sizes satisfy hardware constraints."""
        MC, KC, NC = config.MC, config.KC, config.NC

        # Get MR/NR for kernel
        kernel_dims = {
            "8x8": (8, 8),
            "6x16": (6, 16),
            "4x24": (4, 24),
            "6x16_asm": (6, 16),
        }
        MR, NR = kernel_dims.get(config.kernel, (6, 16))

        # Must be multiples of micro-kernel dimensions
        if MC % MR != 0:
            return False
        if NC % NR != 0:
            return False

        # Reasonable bounds
        if MC < MR or MC > 1024:
            return False
        if KC < 64 or KC > 2048:
            return False
        if NC < NR or NC > 32768:
            return False

        # Cache hierarchy constraints (Haswell)
        dtype_size = 4  # float32
        L2_size = 256 * 1024    # 256 KiB
        L3_size = 25 * 1024 * 1024  # 25 MiB

        a_panel = MC * KC * dtype_size
        b_panel = KC * NC * dtype_size

        if a_panel > L2_size * 2:
            return False
        if b_panel > L3_size:
            return False

        return True


def main():
    """CLI interface for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Generic SGEMM benchmark runner"
    )
    parser.add_argument('--M', type=int, default=1024)
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--K', type=int, default=None)
    parser.add_argument('--kernel', default='6x16')
    parser.add_argument('--parallel', default='2D')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--MC', type=int, default=120)
    parser.add_argument('--KC', type=int, default=512)
    parser.add_argument('--NC', type=int, default=4096)
    parser.add_argument('--r-tasks', type=int, default=1)
    parser.add_argument('--min-runs', type=int, default=3)
    parser.add_argument('--max-runs', type=int, default=15)
    parser.add_argument('--target-stddev', type=float, default=2.0)
    parser.add_argument('--taskset', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    config = SGEMMConfig(
        M=args.M, N=args.N, K=args.K,
        kernel=args.kernel, parallel=args.parallel,
        threads=args.threads,
        MC=args.MC, KC=args.KC, NC=args.NC,
        r_tasks=args.r_tasks
    )

    runner = BenchmarkRunner(
        min_runs=args.min_runs,
        max_runs=args.max_runs,
        target_stddev=args.target_stddev,
        use_taskset=args.taskset,
        verbose=args.verbose
    )

    print(f"Benchmarking: {config.M}×{config.N}×{config.K} "
          f"kernel={config.kernel} threads={config.threads}")
    print(f"Block sizes: MC={config.MC} KC={config.KC} NC={config.NC}")
    print()

    result = runner.run(config)

    if result.success:
        print(f"Results ({result.runs} runs, stddev={result.stddev_pct:.2f}%):")
        print(f"  Median: {result.median_gflops:.2f} GFLOP/s ({result.median_s*1e6:.1f} µs)")
        print(f"  Mean:   {result.mean_gflops:.2f} GFLOP/s ({result.mean_s*1e6:.1f} µs)")
        print(f"  Min:    {result.min_gflops:.2f} GFLOP/s")
        print(f"  Max:    {result.max_gflops:.2f} GFLOP/s")
        if result.all_times:
            print(f"\n  All samples (µs): {[f'{t*1e6:.1f}' for t in result.all_times]}")
    else:
        print(f"FAILED: {result.error}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
