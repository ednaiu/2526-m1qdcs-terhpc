# SGEMM Comparison Report (Direct OpenBLAS vs Week2)

## Scope
- Baseline: direct CBLAS/OpenBLAS run before TASK1 packing optimization (5 runs, median).
- Final: direct CBLAS/OpenBLAS run after optimization (5 runs, median).
- Sizes: 256^3, 512^3, 1024^3.
- Threads: 8 and 16.

## Final Result
- After optimization, Week2 is faster than OpenBLAS in 6/6 tested cases.

## Before vs After (Direct CBLAS)

| Size | Threads | OpenBLAS Before | Week2 Before | Ratio Before | OpenBLAS After | Week2 After | Ratio After | Week2 Gain | Ratio Gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256^3 | 8 | 202.2255 | 152.4961 | 0.7541 | 178.3043 | 216.2483 | 1.2128 | +41.8% | +60.8% |
| 256^3 | 16 | 313.8916 | 167.6121 | 0.5340 | 165.6413 | 217.9581 | 1.3158 | +30.0% | +146.4% |
| 512^3 | 8 | 243.1575 | 176.5301 | 0.7260 | 245.9376 | 373.2563 | 1.5177 | +111.4% | +109.0% |
| 512^3 | 16 | 402.2435 | 253.5758 | 0.6304 | 396.3237 | 485.3528 | 1.2246 | +91.4% | +94.3% |
| 1024^3 | 8 | 272.7594 | 184.1281 | 0.6751 | 271.6239 | 385.6335 | 1.4197 | +109.4% | +110.3% |
| 1024^3 | 16 | 480.9909 | 280.8224 | 0.5838 | 482.4526 | 568.1538 | 1.1776 | +102.3% | +101.7% |

## Notes on Methodology
- Direct benchmark is used (C program calling cblas_sgemm), so results are not affected by NumPy BLAS backend behavior.
- CPU affinity and thread count are controlled in the direct benchmark runner.
- Reported metric is median GFLOP/s across 5 runs per case.

## Source Data Files
- results/direct_openblas_vs_week2.csv (final after optimization).
- Terminal baseline run used for before-optimization direct numbers.
- results/openblas_vs_week2_median30_8_16.csv (older NumPy-based run, not used for final verdict).