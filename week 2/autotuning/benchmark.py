#!/usr/bin/env python3
"""
benchmark.py — Data collection for SGEMM autotuning

Runs the benchmark binary and collects performance data across:
  - Problem sizes (matrix dimensions)
  - Block sizes (MC, KC, NC)
  - Kernel types (8x8, 6x16, 4x24, 6x16_asm)
  - Parallelism modes (2D, 3D)
  - Thread counts

Output: CSV file with timing and GFLOP/s data for subsequent optimization.
"""

import subprocess
import csv
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Configuration
BENCH_BIN = "./bin/bench_sgemm"
RESULTS_DIR = "./results"
KERNELS = ["8x8", "6x16", "4x24", "6x16_asm"]
PARALLEL_MODES = ["2D", "3D"]

# Default problem sizes (in ascending order of complexity)
DEFAULT_SIZES = {
    "small": [(64, 64, 64), (128, 128, 128), (256, 256, 256)],
    "medium": [(512, 512, 512), (1024, 1024, 1024)],
    "large": [(2048, 2048, 2048)],
}

# Default block size configurations to test
DEFAULT_BLOCK_CONFIGS = [
    {"MC": 120, "KC": 512, "NC": 4096},
    {"MC": 96, "KC": 256, "NC": 2048},
    {"MC": 120, "KC": 768, "NC": 4096},
    {"MC": 240, "KC": 512, "NC": 8192},
]

# Default thread counts to test
DEFAULT_THREADS = [1, 2, 4, 8]


def run_benchmark(verbose=False):
    """
    Run the benchmark binary and capture output.
    
    Returns:
        List of dictionaries parsed from CSV output
    """
    if not os.path.exists(BENCH_BIN):
        print(f"ERROR: Benchmark binary not found: {BENCH_BIN}", file=sys.stderr)
        print("Please run 'make bench' first.", file=sys.stderr)
        return None

    try:
        result = subprocess.run(
            [BENCH_BIN],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"ERROR: Benchmark exited with code {result.returncode}", file=sys.stderr)
            if result.stderr:
                print(f"STDERR: {result.stderr}", file=sys.stderr)
            return None
        
        # Parse CSV output
        lines = result.stdout.strip().split('\n')
        if not lines:
            print("ERROR: No output from benchmark", file=sys.stderr)
            return None
        
        # Parse header and rows
        reader = csv.DictReader(lines)
        data = list(reader)
        
        if verbose:
            print(f"Collected {len(data)} benchmark records")
        
        return data
        
    except subprocess.TimeoutExpired:
        print("ERROR: Benchmark timed out", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR: Failed to run benchmark: {e}", file=sys.stderr)
        return None


def save_results(data, filename):
    """Save benchmark results to CSV file."""
    if not data:
        print("ERROR: No data to save", file=sys.stderr)
        return False
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, filename)
    
    try:
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        print(f"Results saved to: {filepath}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to save results: {e}", file=sys.stderr)
        return False


def filter_data(data, **kwargs):
    """
    Filter benchmark data by specified criteria.
    
    Args:
        data: List of benchmark records
        **kwargs: Filter criteria (e.g., kernel="6x16", threads=4)
    
    Returns:
        Filtered list of records
    """
    filtered = data
    for key, value in kwargs.items():
        filtered = [r for r in filtered if str(r.get(key)) == str(value)]
    return filtered


def analyze_results(data):
    """
    Analyze benchmark results and print summary statistics.
    
    Args:
        data: List of benchmark records
    """
    if not data:
        print("No data to analyze", file=sys.stderr)
        return
    
    print("\n" + "="*60)
    print("BENCHMARK ANALYSIS SUMMARY")
    print("="*60)
    
    # Group by kernel
    print("\nPerformance by Kernel Type:")
    print("-" * 60)
    for kernel in KERNELS:
        kernel_data = filter_data(data, kernel=kernel)
        if kernel_data:
            gflops_mean = [float(r.get('gflops_mean', 0)) for r in kernel_data if r.get('gflops_mean')]
            avg_gflops = sum(gflops_mean) / len(gflops_mean) if gflops_mean else 0
            max_gflops = max(gflops_mean) if gflops_mean else 0
            print(f"  {kernel:12s}: avg={avg_gflops:7.1f} GFLOP/s, max={max_gflops:7.1f} GFLOP/s ({len(kernel_data)} runs)")
    
    # Group by problem size
    print("\nPerformance by Problem Size:")
    print("-" * 60)
    for size in ["small", "medium", "large"]:
        size_data = []
        for r in data:
            m, n, k = int(r.get('M', 0)), int(r.get('N', 0)), int(r.get('K', 0))
            if m == n == k:  # Square matrices only
                if size == "small" and m <= 256:
                    size_data.append(r)
                elif size == "medium" and 256 < m <= 1024:
                    size_data.append(r)
                elif size == "large" and m > 1024:
                    size_data.append(r)
        
        if size_data:
            gflops_mean = [float(r.get('gflops_mean', 0)) for r in size_data if r.get('gflops_mean')]
            avg_gflops = sum(gflops_mean) / len(gflops_mean) if gflops_mean else 0
            max_gflops = max(gflops_mean) if gflops_mean else 0
            print(f"  {size:12s}: avg={avg_gflops:7.1f} GFLOP/s, max={max_gflops:7.1f} GFLOP/s ({len(size_data)} runs)")
    
    # Thread scaling
    print("\nThread Scaling (6x16 kernel):")
    print("-" * 60)
    thread_data = filter_data(data, kernel="6x16", parallel="2D")
    if thread_data:
        thread_counts = sorted(set(int(r.get('threads', 1)) for r in thread_data))
        for tc in thread_counts:
            tc_data = filter_data(thread_data, threads=str(tc))
            if tc_data:
                gflops_mean = [float(r.get('gflops_mean', 0)) for r in tc_data if r.get('gflops_mean')]
                avg_gflops = sum(gflops_mean) / len(gflops_mean) if gflops_mean else 0
                print(f"  {tc:2d} threads: avg={avg_gflops:7.1f} GFLOP/s ({len(tc_data)} runs)")
    
    # Best performing configuration
    print("\nTop 5 Best Performing Configurations:")
    print("-" * 60)
    sorted_data = sorted(data, key=lambda r: float(r.get('gflops_mean', 0)), reverse=True)
    for i, r in enumerate(sorted_data[:5], 1):
        print(f"  {i}. M={r.get('M'):5s} N={r.get('N'):5s} K={r.get('K'):5s} "
              f"{r.get('kernel'):8s} {r.get('parallel'):3s} "
              f"threads={r.get('threads'):2s} → {float(r.get('gflops_mean', 0)):7.1f} GFLOP/s")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Collect benchmark data for SGEMM autotuning"
    )
    parser.add_argument('-o', '--output', default=None,
                        help='Output filename (default: auto-generated with timestamp)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('-a', '--analyze', action='store_true',
                        help='Print analysis after benchmarking')
    
    args = parser.parse_args()
    
    print("SGEMM Benchmark Data Collection")
    print("=" * 60)
    print(f"Benchmark binary: {BENCH_BIN}")
    print(f"Results directory: {RESULTS_DIR}")
    print("")
    
    # Run benchmark
    print("Running benchmark suite...")
    data = run_benchmark(verbose=args.verbose)
    
    if not data:
        print("ERROR: Benchmark failed", file=sys.stderr)
        return 1
    
    # Generate output filename if not specified
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"benchmark_{timestamp}.csv"
    
    # Save results
    if not save_results(data, args.output):
        return 1
    
    # Analyze if requested
    if args.analyze:
        analyze_results(data)
    
    print(f"Benchmark completed successfully: {len(data)} records collected")
    return 0


if __name__ == '__main__':
    sys.exit(main())
