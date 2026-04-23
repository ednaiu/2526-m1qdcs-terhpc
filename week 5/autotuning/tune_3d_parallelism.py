#!/usr/bin/env python3
"""
tune_3d_parallelism.py — Tune TASK2 (3D parallelism) parameters

Sweeps r_tasks values for various problem sizes and compares
TASK1 (2D) vs TASK2 (3D) to find the crossover point where
3D parallelism starts outperforming 2D.

3D parallelism is most useful when:
    - Small matrix + many cores → not enough 2D tiles to saturate all cores
    - r_tasks splits K-dimension to create more parallel work

Usage:
    python3 tune_3d_parallelism.py
    python3 tune_3d_parallelism.py --sizes 128 256 512 --threads 8 16
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_runner import BenchmarkRunner, SGEMMConfig

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def tune(sizes, thread_counts, r_values, kernel, verbose):
    """Compare TASK1 vs TASK2 across configurations."""
    runner = BenchmarkRunner(verbose=verbose, max_runs=10, target_stddev=2.0)
    results = []

    for size in sizes:
        for threads in thread_counts:
            print(f"\n--- {size}×{size}×{size}, {threads} threads ---")

            # TASK1 baseline (2D)
            cfg_2d = SGEMMConfig(
                M=size, N=size, K=size,
                kernel=kernel, parallel="2D",
                threads=threads, MC=120, KC=512, NC=4096
            )
            res_2d = runner.run(cfg_2d, use_cache=False)
            gflops_2d = res_2d.median_gflops if res_2d.success else 0.0
            print(f"  TASK1 (2D):           {gflops_2d:8.2f} GFLOP/s")

            row = {
                'size': size, 'threads': threads,
                '2D': gflops_2d,
            }

            # TASK2 with different r_tasks
            best_3d = 0.0
            best_r = 1
            for r in r_values:
                cfg_3d = SGEMMConfig(
                    M=size, N=size, K=size,
                    kernel=kernel, parallel="3D",
                    threads=threads, MC=120, KC=512, NC=4096,
                    r_tasks=r
                )
                res_3d = runner.run(cfg_3d, use_cache=False)
                gflops_3d = res_3d.median_gflops if res_3d.success else 0.0

                marker = " ★" if gflops_3d > gflops_2d else ""
                print(f"  TASK2 (3D, r={r:2d}):     {gflops_3d:8.2f} GFLOP/s{marker}")

                row[f'3D_r{r}'] = gflops_3d
                if gflops_3d > best_3d:
                    best_3d = gflops_3d
                    best_r = r

            # Summary for this config
            winner = "3D" if best_3d > gflops_2d else "2D"
            ratio = best_3d / gflops_2d if gflops_2d > 0 else 0
            row['winner'] = winner
            row['best_r'] = best_r if winner == "3D" else 0
            row['ratio_3d_over_2d'] = ratio

            if winner == "3D":
                print(f"  → 3D (r={best_r}) wins by {(ratio-1)*100:.1f}%")
            else:
                print(f"  → 2D wins (3D best ratio: {ratio:.2f}x)")

            results.append(row)

    return results


def print_summary(results):
    """Print summary of TASK1 vs TASK2 comparison."""
    print(f"\n{'='*80}")
    print("TASK1 (2D) vs TASK2 (3D) COMPARISON")
    print(f"{'='*80}")

    print(f"\n{'Size':>6s} {'Thr':>4s} {'2D GFLOP/s':>12s} {'Best 3D':>12s} {'Best r':>8s} {'Ratio':>8s} {'Winner':>8s}")
    print("-" * 80)

    for r in results:
        best_3d_vals = [v for k, v in r.items() if k.startswith('3D_r')]
        best_3d = max(best_3d_vals) if best_3d_vals else 0

        print(f"{r['size']:>6d} {r['threads']:>4d} "
              f"{r['2D']:>12.2f} {best_3d:>12.2f} "
              f"{r.get('best_r', '-'):>8} "
              f"{r.get('ratio_3d_over_2d', 0):>8.2f} "
              f"{r['winner']:>8s}")

    print("-" * 80)


def save_results(results):
    """Save results to files."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_file = os.path.join(RESULTS_DIR, "task1_vs_task2.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV saved to: {csv_file}")

    json_file = os.path.join(RESULTS_DIR, "task1_vs_task2.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved to: {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Tune 3D parallelism (TASK2) parameters"
    )
    parser.add_argument('--sizes', type=int, nargs='+', default=[128, 256, 512, 1024],
                        help='Matrix sizes to test')
    parser.add_argument('--threads', type=int, nargs='+', default=[4, 8, 16],
                        help='Thread counts')
    parser.add_argument('--r-values', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='r_tasks values to test')
    parser.add_argument('--kernel', default='6x16')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    print("=" * 70)
    print("TASK1 (2D) vs TASK2 (3D) Parallelism Tuning")
    print("=" * 70)
    print(f"Sizes: {args.sizes}")
    print(f"Threads: {args.threads}")
    print(f"r_tasks values: {args.r_values}")

    results = tune(args.sizes, args.threads, args.r_values,
                   args.kernel, args.verbose)

    print_summary(results)
    save_results(results)

    return 0


if __name__ == '__main__':
    sys.exit(main())
