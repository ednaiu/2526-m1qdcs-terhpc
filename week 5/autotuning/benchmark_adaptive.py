#!/usr/bin/env python3
"""
benchmark_adaptive.py — Adaptive execution timing with median and smart run count

Features:
    - Starts with n=3 executions
    - Increases n if relative stddev > 2%
    - Uses MEDIAN for central tendency (not average)
    - Stops when stddev < 2% or max_runs reached
    - Compatible with existing bench_sgemm.c output
"""

import subprocess
import csv
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import os

BENCH_BIN = "./bin/bench_sgemm"
RESULTS_DIR = "./results"

def parse_bench_output(output_text):
    """Parse CSV from bench_sgemm and return list of dicts"""
    lines = output_text.strip().split('\n')
    reader = csv.DictReader(lines)
    return list(reader)

def run_single_benchmark(M, N, K, kernel="6x16", threads=1):
    """Run single benchmark and return execution time in microseconds"""
    if not os.path.exists(BENCH_BIN):
        print(f"ERROR: {BENCH_BIN} not found. Run 'make bench' first.", file=sys.stderr)
        return None
    
    try:
        result = subprocess.run(
            [BENCH_BIN],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, 'OMP_NUM_THREADS': str(threads)}
        )
        
        if result.returncode != 0:
            print(f"ERROR: Benchmark failed", file=sys.stderr)
            return None
        
        data = parse_bench_output(result.stdout)
        
        # Find matching entry
        for row in data:
            if (int(row['M']) == M and int(row['N']) == N and int(row['K']) == K and
                row['kernel'] == kernel and int(row['threads']) == threads):
                return float(row['time_mean_us'])
        
        return None
    
    except subprocess.TimeoutExpired:
        print(f"ERROR: Benchmark timeout", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return None

def adaptive_timing(M, N, K, kernel="6x16", threads=1,
                   target_stddev=0.02, max_runs=15, verbose=False):
    """
    Collect timing measurements adaptively.
    
    Args:
        M, N, K: Matrix dimensions
        kernel: Kernel type
        threads: Number of threads
        target_stddev: Target relative stddev (2% = 0.02)
        max_runs: Maximum number of runs total
        verbose: Print progress information
    
    Returns:
        dict with statistics
    """
    times = []
    run_count = 0
    
    while run_count < max_runs:
        # Run benchmark
        t = run_single_benchmark(M, N, K, kernel, threads)
        if t is None:
            return None
        
        times.append(t)
        run_count += 1
        
        # Compute statistics so far
        median = np.median(times)
        mean = np.mean(times)
        stddev_abs = np.std(times, ddof=1) if len(times) > 1 else 0
        stddev_rel = stddev_abs / median if median > 0 else 1.0
        
        if verbose:
            print(f"  Run {run_count:2d}: {t:8.2f} µs | "
                  f"median={median:8.2f} µs stddev={stddev_rel*100:5.2f}%", flush=True)
        
        # Check convergence
        if stddev_rel < target_stddev and run_count >= 3:
            if verbose:
                print(f"  ✓ Converged at run {run_count} (stddev={stddev_rel*100:.2f}% < {target_stddev*100:.1f}%)")
            break
    
    # Final statistics
    median = np.median(times)
    mean = np.mean(times)
    stddev_abs = np.std(times, ddof=1) if len(times) > 1 else 0
    stddev_rel = stddev_abs / median if median > 0 else 0
    gflops = (2 * M * N * K) / (median * 1e-6 * 1e9)  # Convert µs to s
    
    return {
        'problem_size': (M, N, K),
        'kernel': kernel,
        'threads': threads,
        'runs': run_count,
        'median_us': median,
        'mean_us': mean,
        'stddev_us': stddev_abs,
        'stddev_pct': stddev_rel * 100,
        'gflops': gflops,
        'all_times': times
    }

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive timing benchmark with median and smart run count"
    )
    parser.add_argument('-m', '--m', type=int, default=1024,
                        help='Matrix dimension M (default: 1024)')
    parser.add_argument('-n', '--n', type=int, default=None,
                        help='Matrix dimension N (default: same as M)')
    parser.add_argument('-k', '--k', type=int, default=None,
                        help='Matrix dimension K (default: same as M)')
    parser.add_argument('--kernel', default='6x16',
                        help='Kernel type (8x8, 6x16, 4x24, 6x16_asm)')
    parser.add_argument('-t', '--threads', type=int, default=1,
                        help='Number of threads (default: 1)')
    parser.add_argument('--target-stddev', type=float, default=0.02,
                        help='Target relative stddev (default: 0.02 = 2%%)')
    parser.add_argument('--max-runs', type=int, default=15,
                        help='Maximum number of runs (default: 15)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output CSV filename')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    M = args.m
    N = args.n or M
    K = args.k or M
    
    print("=" * 70)
    print("Adaptive SGEMM Benchmark with Median Timing")
    print("=" * 70)
    print(f"Problem: {M} × {N} × {K}")
    print(f"Kernel: {args.kernel}, Threads: {args.threads}")
    print(f"Target stddev: {args.target_stddev*100:.1f}%, Max runs: {args.max_runs}")
    print()
    
    # Run adaptive benchmark
    print(f"Starting adaptive timing (estimating execution time)...")
    result = adaptive_timing(M, N, K, args.kernel, args.threads,
                            args.target_stddev, args.max_runs, args.verbose)
    
    if not result:
        print("ERROR: Benchmark failed", file=sys.stderr)
        return 1
    
    # Print results
    print()
    print("=" * 70)
    print("RESULTS WITH ADAPTIVE EXECUTION")
    print("=" * 70)
    print(f"Total runs: {result['runs']}")
    print(f"Median time: {result['median_us']:.2f} µs")
    print(f"Mean time:   {result['mean_us']:.2f} µs")
    print(f"Std. Dev:    {result['stddev_us']:.2f} µs ({result['stddev_pct']:.2f}%)")
    print(f"Performance: {result['gflops']:.2f} GFLOP/s")
    print()
    print("All timing samples (microseconds):")
    for i, t in enumerate(result['all_times'], 1):
        marker = " ← MEDIAN" if t == result['median_us'] else ""
        print(f"  Run {i:2d}: {t:8.2f} µs{marker}")
    
    # Save to CSV if requested
    if args.output:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        filepath = os.path.join(RESULTS_DIR, args.output)
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'M', 'N', 'K', 'kernel', 'threads',
                'runs', 'median_us', 'mean_us', 'stddev_us', 'stddev_pct', 'gflops'
            ])
            writer.writeheader()
            writer.writerow({
                'M': M, 'N': N, 'K': K,
                'kernel': args.kernel,
                'threads': args.threads,
                'runs': result['runs'],
                'median_us': f"{result['median_us']:.2f}",
                'mean_us': f"{result['mean_us']:.2f}",
                'stddev_us': f"{result['stddev_us']:.2f}",
                'stddev_pct': f"{result['stddev_pct']:.2f}",
                'gflops': f"{result['gflops']:.2f}"
            })
        print(f"\nResults saved to: {filepath}")
    
    print("=" * 70)
    return 0

if __name__ == '__main__':
    sys.exit(main())
