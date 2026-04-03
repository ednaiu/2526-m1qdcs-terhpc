#!/usr/bin/env python3
"""
compare_with_openblas.py — Performance comparison: Week 2 SGEMM vs OpenBLAS

Compares SGEMM implementations on various matrix sizes and thread counts.
Requires: NumPy (uses OpenBLAS backend), scipy (optional for advanced analysis)

Usage:
    python3 compare_with_openblas.py                    # Default matrices
    python3 compare_with_openblas.py --sizes 256 512 1024  # Custom sizes
    python3 compare_with_openblas.py -t 1 2 4 8         # Thread scaling
    python3 compare_with_openblas.py --output results.csv  # Save results
"""

import sys
import os
import csv
import argparse
import subprocess
import time
from pathlib import Path

import numpy as np

BENCH_BIN = "./bin/bench_sgemm"
RESULTS_DIR = "./results"

# Ensure OpenBLAS is being used by NumPy
print("NumPy BLAS backend information:")
try:
    import numpy
    print(f"  NumPy version: {numpy.__version__}")
    # Try to get BLAS info
    try:
        numpy.show_config()
    except:
        print("  (BLAS info not available in this NumPy version)")
except:
    pass

class OpenBLASBenchmark:
    """Benchmark wrapper for OpenBLAS via NumPy"""
    
    @staticmethod
    def sgemm(M, N, K, threads=1, n_runs=5):
        """
        Benchmark matrix multiplication using NumPy (OpenBLAS backend).
        
        Args:
            M, N, K: Dimensions
            threads: Number of threads
            n_runs: Number of repetitions
        
        Returns:
            dict with timing and GFLOP/s
        """
        # Set thread count
        os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
        os.environ['MKL_NUM_THREADS'] = str(threads)
        os.environ['OMP_NUM_THREADS'] = str(threads)
        
        # Generate data
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        # Warm-up
        _ = A @ B
        
        # Time runs
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            C = A @ B
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        median_time = np.median(times)
        mean_time = np.mean(times)
        stddev = np.std(times)
        gflops = (2 * M * N * K) / (median_time * 1e9) if median_time > 0 else 0
        
        return {
            'M': M, 'N': N, 'K': K, 'threads': threads,
            'time_median_s': median_time,
            'time_mean_s': mean_time,
            'time_stddev_s': stddev,
            'gflops': gflops,
            'runs': n_runs
        }


class Week2Benchmark:
    """Benchmark wrapper for Week 2 SGEMM"""
    
    @staticmethod
    def parse_bench_output(output_text):
        """Parse CSV output from bench_sgemm"""
        lines = [line.strip() for line in output_text.strip().split('\n') if line.strip()]
        if not lines:
            return []
        
        import csv as csv_module
        try:
            header_idx = None
            for i, line in enumerate(lines):
                if line.startswith("M,N,K,kernel,parallel,threads,"):
                    header_idx = i
                    break

            if header_idx is None:
                return []

            csv_lines = [lines[header_idx]]
            expected_cols = lines[header_idx].count(',') + 1
            for line in lines[header_idx + 1:]:
                if line.startswith("Testing ") or line.startswith("Benchmark completed"):
                    continue
                if line.count(',') + 1 == expected_cols:
                    csv_lines.append(line)

            reader = csv_module.DictReader(csv_lines)
            return list(reader)
        except:
            return []
    
    @staticmethod
    def sgemm(M, N, K, kernel='6x16', threads=1):
        """
        Run Week 2 SGEMM benchmark.
        
        Args:
            M, N, K: Dimensions
            kernel: Kernel type (6x16 recommended)
            threads: Number of threads
        
        Returns:
            dict with timing and GFLOP/s
        """
        if not os.path.exists(BENCH_BIN):
            print(f"ERROR: {BENCH_BIN} not found. Run 'make bench' first.", file=sys.stderr)
            return None
        
        try:
            # Set threads
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = str(threads)
            
            result = subprocess.run(
                [BENCH_BIN],
                capture_output=True,
                text=True,
                timeout=120,
                env=env
            )
            
            if result.returncode != 0:
                print(f"ERROR: Benchmark failed: {result.stderr}", file=sys.stderr)
                return None
            
            data = Week2Benchmark.parse_bench_output(result.stdout)
            
            # Find matching entry (first one matching size and threads)
            for row in data:
                if (int(row.get('M', 0)) == M and 
                    int(row.get('N', 0)) == N and 
                    int(row.get('K', 0)) == K and
                    int(row.get('threads', 0)) == threads and
                    row.get('kernel', '') == kernel):
                    
                    time_us = float(row.get('time_mean_us', 0))
                    time_s = time_us / 1e6
                    gflops = float(row.get('gflops_mean', 0))
                    
                    return {
                        'M': M, 'N': N, 'K': K, 'kernel': kernel, 'threads': threads,
                        'time_s': time_s,
                        'gflops': gflops
                    }
            
            print(f"WARNING: No matching benchmark for {M}×{N}×{K}", file=sys.stderr)
            return None
        
        except subprocess.TimeoutExpired:
            print(f"ERROR: Benchmark timeout for {M}×{N}×{K}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return None


def compare(M, N, K, threads=1, kernel='6x16'):
    """Compare OpenBLAS vs Week 2 for single configuration"""
    print(f"\n--- {M} × {N} × {K}, {threads} thread{'s' if threads != 1 else ''} ---")
    
    # OpenBLAS benchmark
    print(f"  Running OpenBLAS...", end='', flush=True)
    ob = OpenBLASBenchmark.sgemm(M, N, K, threads, n_runs=3)
    if ob:
        print(f" {ob['gflops']:.1f} GFLOP/s")
    else:
        print(" FAILED")
        ob = None
    
    # Week 2 benchmark
    print(f"  Running Week 2 SGEMM...", end='', flush=True)
    w2 = Week2Benchmark.sgemm(M, N, K, kernel, threads)
    if w2:
        print(f" {w2['gflops']:.1f} GFLOP/s")
    else:
        print(" FAILED")
        w2 = None
    
    # Compare
    if ob and w2:
        ratio = w2['gflops'] / ob['gflops']
        marker = "✓" if ratio >= 0.9 else "⚠" if ratio >= 0.75 else "✗"
        print(f"  {marker} Week 2 is {ratio*100:.1f}% of OpenBLAS")
        
        return {
            'M': M, 'N': N, 'K': K, 'threads': threads,
            'kernel': kernel,
            'openblas_gflops': ob['gflops'],
            'week2_gflops': w2['gflops'],
            'ratio': ratio,
            'status': marker
        }
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare Week 2 SGEMM with OpenBLAS performance"
    )
    parser.add_argument('--sizes', type=int, nargs='+', default=[256, 512, 1024],
                        help='Matrix sizes to test (default: 256 512 1024)')
    parser.add_argument('-t', '--threads', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Thread counts to test (default: 1 2 4 8)')
    parser.add_argument('--kernel', default='6x16',
                        help='Week 2 kernel type (default: 6x16)')
    parser.add_argument('-o', '--output', default=None,
                        help='Save results to CSV')
    parser.add_argument('--skip-openblas', action='store_true',
                        help='Skip OpenBLAS benchmarks (Week 2 only)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SGEMM Performance Comparison: Week 2 vs OpenBLAS")
    print("=" * 70)
    print(f"Matrix sizes: {args.sizes}")
    print(f"Thread counts: {args.threads}")
    print(f"Week 2 kernel: {args.kernel}")
    if args.skip_openblas:
        print("(OpenBLAS benchmarks skipped)")
    print()
    
    results = []
    
    for size in args.sizes:
        for threads in args.threads:
            if args.skip_openblas:
                # Week 2 only
                w2 = Week2Benchmark.sgemm(size, size, size, args.kernel, threads)
                if w2:
                    print(f"{size}×{size}×{size} on {threads}T: {w2['gflops']:.1f} GFLOP/s (Week 2)")
                    results.append({
                        'M': size, 'N': size, 'K': size, 'threads': threads,
                        'week2_gflops': w2['gflops'],
                        'ratio': 'N/A'
                    })
            else:
                # Full comparison
                row = compare(size, size, size, threads, args.kernel)
                if row:
                    results.append(row)
    
    # Summary table
    if results and not args.skip_openblas:
        print()
        print("=" * 70)
        print("SUMMARY TABLE")
        print("=" * 70)
        print(f"{'Size':10s} {'Threads':8s} {'OpenBLAS':>12s} {'Week 2':>12s} {'Ratio':>10s} {'Status'}")
        print("-" * 70)
        for r in results:
            size = f"{r['M']}³"
            print(f"{size:10s} {r['threads']:8d} {r['openblas_gflops']:12.1f} "
                  f"{r['week2_gflops']:12.1f} {r['ratio']:10.1%} {r['status']:>6s}")
    
    # Save results
    if args.output and results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        filepath = os.path.join(RESULTS_DIR, args.output)
        with open(filepath, 'w', newline='') as f:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {filepath}")
    
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
