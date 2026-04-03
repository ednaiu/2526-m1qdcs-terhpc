#!/usr/bin/env python3
"""
kernel_comparison.py — Compare all micro-kernel variants across problem sizes

Benchmarks all 4 kernels (8x8, 6x16, 4x24, 6x16_asm) for each problem size
and recommends the most efficient kernel per size.

Uses adaptive N timing with median for reliable results.

Usage:
    python3 kernel_comparison.py
    python3 kernel_comparison.py --sizes 256 512 1024 2048 --threads 1 4 8
    python3 kernel_comparison.py -v
"""

import os
import sys
import json
import argparse
import csv
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_runner import BenchmarkRunner, SGEMMConfig

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

KERNELS = ["8x8", "6x16", "4x24", "6x16_asm"]

# Known-good block sizes per kernel
DEFAULT_BLOCK_SIZES = {
    "8x8":      {"MC": 120, "KC": 512, "NC": 4096},  # MC must be mult of 8
    "6x16":     {"MC": 120, "KC": 512, "NC": 4096},  # MC mult of 6
    "4x24":     {"MC": 120, "KC": 512, "NC": 4080},  # MC mult of 4, NC mult of 24
    "6x16_asm": {"MC": 120, "KC": 512, "NC": 4096},  # Same as 6x16
}

# Fix block sizes to be proper multiples
def fix_block_sizes(kernel: str, MC: int, KC: int, NC: int) -> tuple:
    """Ensure MC and NC are multiples of MR and NR."""
    dims = {"8x8": (8, 8), "6x16": (6, 16), "4x24": (4, 24), "6x16_asm": (6, 16)}
    MR, NR = dims.get(kernel, (6, 16))

    MC = MR * (MC // MR) if MC >= MR else MR
    NC = NR * (NC // NR) if NC >= NR else NR
    return MC, KC, NC


def run_comparison(sizes: list, thread_counts: list, verbose: bool):
    """Run kernel comparison across all sizes and thread counts."""
    runner = BenchmarkRunner(verbose=verbose, max_runs=10, target_stddev=2.0)
    results = []

    for size in sizes:
        for threads in thread_counts:
            print(f"\n--- {size}×{size}×{size}, {threads} thread{'s' if threads > 1 else ''} ---")

            row = {'size': size, 'threads': threads}

            for kernel in KERNELS:
                bs = DEFAULT_BLOCK_SIZES[kernel]
                MC, KC, NC = fix_block_sizes(kernel, bs['MC'], bs['KC'], bs['NC'])

                config = SGEMMConfig(
                    M=size, N=size, K=size,
                    kernel=kernel, threads=threads,
                    MC=MC, KC=KC, NC=NC
                )

                if not runner.is_valid_config(config):
                    print(f"  {kernel:10s}: SKIPPED (invalid block sizes)")
                    row[kernel] = 0.0
                    continue

                result = runner.run(config, use_cache=False)

                if result.success:
                    gflops = result.median_gflops
                    print(f"  {kernel:10s}: {gflops:8.2f} GFLOP/s "
                          f"(median of {result.runs}, stddev={result.stddev_pct:.1f}%)")
                    row[kernel] = gflops
                else:
                    print(f"  {kernel:10s}: FAILED ({result.error})")
                    row[kernel] = 0.0

            # Determine winner
            best_kernel = max(KERNELS, key=lambda k: row.get(k, 0))
            best_gflops = row.get(best_kernel, 0)
            row['winner'] = best_kernel
            row['best_gflops'] = best_gflops
            print(f"  → Winner: {best_kernel} ({best_gflops:.2f} GFLOP/s)")

            results.append(row)

    return results


def print_summary(results: list):
    """Print formatted summary table."""
    print(f"\n{'='*90}")
    print("KERNEL COMPARISON SUMMARY")
    print(f"{'='*90}")

    print(f"\n{'Size':>6s} {'Thr':>4s} ", end="")
    for k in KERNELS:
        print(f"{'  ' + k:>12s}", end="")
    print(f"  {'Winner':>10s}")
    print("-" * 90)

    for r in results:
        print(f"{r['size']:>6d} {r['threads']:>4d} ", end="")
        for k in KERNELS:
            g = r.get(k, 0)
            marker = " ★" if k == r['winner'] else "  "
            print(f"{g:>10.2f}{marker}", end="")
        print(f"  {r['winner']:>10s}")

    print("-" * 90)


def save_results(results: list):
    """Save results to CSV and JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # CSV
    csv_file = os.path.join(RESULTS_DIR, "kernel_comparison.csv")
    fieldnames = ['size', 'threads'] + KERNELS + ['winner', 'best_gflops']
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved to: {csv_file}")

    # JSON summary
    json_file = os.path.join(RESULTS_DIR, "kernel_comparison.json")
    with open(json_file, 'w') as f:
        json.dump({
            'kernels_tested': KERNELS,
            'results': results,
            'recommendation': _compute_recommendation(results),
        }, f, indent=2)
    print(f"JSON saved to: {json_file}")


def _compute_recommendation(results: list) -> dict:
    """Compute per-size kernel recommendation."""
    recs = {}
    for r in results:
        key = f"{r['size']}x{r['size']}_t{r['threads']}"
        recs[key] = {
            'kernel': r['winner'],
            'gflops': r['best_gflops'],
        }
    return recs


def main():
    parser = argparse.ArgumentParser(
        description="Compare all SGEMM micro-kernel variants"
    )
    parser.add_argument('--sizes', type=int, nargs='+', default=[128, 256, 512, 1024],
                        help='Matrix sizes to test')
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 4, 8],
                        help='Thread counts to test')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    print("=" * 70)
    print("SGEMM Micro-Kernel Comparison")
    print("=" * 70)
    print(f"Kernels: {', '.join(KERNELS)}")
    print(f"Sizes: {args.sizes}")
    print(f"Threads: {args.threads}")

    results = run_comparison(args.sizes, args.threads, args.verbose)
    print_summary(results)
    save_results(results)

    return 0


if __name__ == '__main__':
    sys.exit(main())
