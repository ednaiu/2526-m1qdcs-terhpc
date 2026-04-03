# Week 2 Slide Content (Paste-Ready for Figma)

## Slide 1
Title: Week 2 HPC: High-Performance SGEMM
Subtitle: Optimized matrix multiplication on x86-64 with AVX2+FMA
Metrics:
- Peak throughput: 781 GF/s
- Best 8-thread ratio: 182% vs OpenBLAS (1024^3)
- Best 16-thread ratio: 151% vs OpenBLAS (2048^3)

## Slide 2
Title: Problem Statement
Equation: C = alpha * A * B + beta * C
Body:
- Naive cost is 2N^3 FLOPs
- Build a hand-optimized SGEMM in C
- Combine SIMD, cache blocking, and OpenMP
- Outperform OpenBLAS on multicore hardware

## Slide 3
Title: Why SGEMM Is Hard
Columns:
- CPU throughput: keep AVX2+FMA units fully utilized
- Cache hierarchy: reuse data from L1/L2/L3, avoid DRAM stalls
- Parallelism: maximize thread utilization with low synchronization overhead
Callout: Top performance appears only when all three are optimized together.

## Slide 4
Title: BLIS 5-Loop Architecture
Flow:
jc (NC, L3) -> pc (KC, L2) -> pack B -> ic (MC, L2) -> pack A -> jr (NR) -> ir (MR) -> micro-kernel
Notes:
- Packing transforms strided loads into contiguous streams
- Better cache and TLB behavior
Blocking:
- MC = 120
- KC = 512
- NC = 4096

## Slide 5
Title: Micro-Kernel Comparison
Table rows:
- 8x8: 10/16 regs, 128 FLOP/step, simple but underutilized
- 6x16: 16/16 regs, 192 FLOP/step, best default
- 4x24: 16/16 regs, 192 FLOP/step, good for wide-N
- 6x16 ASM: 16/16 regs, 192 FLOP/step, +5-15% vs intrinsics on large sizes

## Slide 6
Title: TASK1 (2D Parallelism)
Body:
- Parallelize over tile space (ic, jt)
- Parallel pack B
- Pre-pack A once per (pc, ic), then reuse
- Compute with collapse(2) + dynamic scheduling
Impact:
- Removed repeated A packing
- Reduced overhead
- Up to ~2x speedup vs early Week 2 implementation

## Slide 7
Title: TASK2 (3D Parallelism)
Body:
- Split K into r slices
- Each task computes private partial C
- Reduce partials into final C with AVX2
Pros:
- More task-level parallelism when MxN is small
Limits:
- Reduction after taskwait is a bottleneck
- Extra memory traffic from partial buffers
- Usually slower than TASK1 for square matrices

## Slide 8
Title: Auto-Tuning Pipeline
Methods:
- Gradient Descent (coordinate descent)
- Bayesian Optimization (Optuna TPE)
Protocol:
- Adaptive-N timing
- Stop when stddev/median < 2%
- JSON interface between Python and C benchmark
Outcome:
- Stable machine-specific blocking parameters
- Better than manual tuning

## Slide 9
Title: Five Critical Bugs and Fixes
1. Edge tile OOB writes -> boundary scratch buffer
2. malloc/free inside tasks -> thread-local buffers outside hot path
3. gflops_min/max swapped -> corrected time-to-throughput mapping
4. Python/C interface mismatch -> single-config JSON CLI mode
5. Repeated A packing in TASK1 -> pack once, reuse across jt strips

## Slide 10
Title: Results at 8 Threads vs OpenBLAS
Rows:
- 256^3: 221.9 -> 247.0 GF/s (111%)
- 512^3: 252.0 -> 384.2 GF/s (152%)
- 1024^3: 272.2 -> 495.0 GF/s (182%)
- 2048^3: 558.2 -> 459.3 GF/s (82%)
Callout:
- Peak at 8T: 495 GF/s, 182% at 1024^3

## Slide 11
Title: Results at 16 Threads vs OpenBLAS
Rows:
- 512^3: 424.7 -> 510.6 GF/s (120%)
- 1024^3: 486.8 -> 659.3 GF/s (135%)
- 2048^3: 517.3 -> 781.0 GF/s (151%)
Callout:
- Peak at 16T: 781 GF/s, 151% at 2048^3

## Slide 12
Title: Scaling from 8T to 16T
Comparison (6x16 TASK1):
- 256^3: ours x1.13, OpenBLAS x1.43
- 512^3: ours x1.40, OpenBLAS x1.69
- 1024^3: ours x1.34, OpenBLAS x1.79
- 2048^3: ours x1.67, OpenBLAS x0.93
Key message:
- At 2048^3, OpenBLAS scales down while our implementation scales up.

## Slide 13
Title: Conclusion and Next Steps
Done:
- 4 micro-kernels including ASM
- BLIS 5-loop with packing
- TASK1 and TASK2
- Auto-tuning pipeline
- 96/96 correctness tests passed
Next steps:
- Parallel TASK2 reduction
- Small-size fast path
- NUMA-aware pinning and scheduler tuning
Closing line:
- Architecture-level decisions delivered the performance gains.
