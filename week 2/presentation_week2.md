---
title: Week 2 HPC: High-Performance SGEMM
author: TER HPC Project
date: April 2026
---

# Week 2 HPC: High-Performance SGEMM

- Course: TER HPC
- Topic: Optimized SGEMM on x86-64 (AVX2+FMA)
- Goal: Beat OpenBLAS on multicore CPU

---

# 1) Problem Statement

SGEMM:

$$
C = \alpha AB + \beta C
$$

- Naive complexity: $2N^3$ FLOPs
- Project goal:
1. Build a hand-optimized SGEMM in C
2. Use SIMD, cache blocking, and OpenMP
3. Outperform OpenBLAS on a multicore machine

---

# 2) Why This Is Hard

Three bottlenecks must be solved at the same time:

1. CPU throughput
- Keep AVX2+FMA pipelines busy

2. Cache and memory hierarchy
- Minimize DRAM traffic, maximize L1/L2/L3 reuse

3. Parallelism
- Distribute work across threads with minimal idle time

Core point: optimizing only one axis does not deliver top performance.

---

# 3) Architecture: BLIS 5-Loop

```
jc (NC, L3)
  pc (KC, L2)
    pack B
    ic (MC, L2)
      pack A
      jr (NR, registers)
        ir (MR, registers)
          micro-kernel
```

- Packing A and B turns memory access into sequential streams
- This reduces cache misses and TLB pressure

Auto-tuned blocking parameters:

- MC = 120
- KC = 512
- NC = 4096

---

# 4) Micro-Kernels

| Kernel | Tile (MRxNR) | Registers | FLOP/step | Best use case |
|---|---:|---:|---:|---|
| 8x8 | 8x8 | 10/16 | 128 | Simple and robust, but under-utilizes register file |
| 6x16 | 6x16 | 16/16 | 192 | Default kernel, best overall balance |
| 4x24 | 4x24 | 16/16 | 192 | Good for wide-N workloads |
| 6x16 ASM | 6x16 | 16/16 | 192 | Best peak on large sizes, +5-15% vs intrinsics |

---

# 5) Parallelism TASK1 (2D)

Idea: parallelize over the tile space $(ic, jt)$.

- Parallel pack B
- Pre-pack A once per $(pc, ic)$ instead of once per $jt$
- Compute: collapse(2) + dynamic scheduling

Impact:

- Eliminated repeated A packing
- Reduced packing overhead
- Up to ~2x speedup vs early Week 2 implementation

---

# 6) Parallelism TASK2 (3D)

Idea: split K into $r$ slices, compute private partial C buffers, then reduce.

- Each task computes one K-slice
- Final reduction into C is AVX2-vectorized

Benefits:

- More task-level parallelism when MxN is small

Limitations:

- Post-taskwait reduction becomes a bottleneck
- Extra memory traffic from partial C buffers
- Usually slower than TASK1 for square matrices

---

# 7) Auto-Tuning

Methods used:

1. Gradient Descent (coordinate descent)
2. Bayesian Optimization (Optuna TPE)

Benchmark protocol:

- Adaptive-N timing
- Stop criterion: $\sigma/\text{median} < 2\%$
- JSON interface for Python tuner scripts

Outcome:

- Finds stable machine-specific blocking parameters
- Best configs outperform manual tuning

---

# 8) Five Critical Bugs and Fixes

1. Edge tile OOB write
- Fix: boundary scratch buffer with gather, zero-pad, scatter

2. malloc/free inside OpenMP tasks
- Fix: thread-local buffers, allocate outside hot path

3. Swapped gflops_min and gflops_max
- Fix: correct mapping from min/max time to throughput

4. Python tuner could not parse benchmark output
- Fix: single-config CLI mode with JSON output

5. Repeated A packing in TASK1
- Fix: pack once and reuse across jt strips

---

# 9) Results: 8 Threads vs OpenBLAS

| Size | OpenBLAS | 6x16 TASK1 | Ratio |
|---:|---:|---:|---:|
| 256^3 | 221.9 GF/s | 247.0 GF/s | 111% |
| 512^3 | 252.0 GF/s | 384.2 GF/s | 152% |
| 1024^3 | 272.2 GF/s | 495.0 GF/s | 182% |
| 2048^3 | 558.2 GF/s | 459.3 GF/s | 82% |

Peak at 8T: 495 GF/s = 182% of OpenBLAS (1024^3).

---

# 10) Results: 16 Threads vs OpenBLAS

| Size | OpenBLAS | 6x16 ASM | Ratio |
|---:|---:|---:|---:|
| 512^3 | 424.7 GF/s | 510.6 GF/s | 120% |
| 1024^3 | 486.8 GF/s | 659.3 GF/s | 135% |
| 2048^3 | 517.3 GF/s | 781.0 GF/s | 151% |

Peak at 16T: 781 GF/s = 151% of OpenBLAS (2048^3).

---

# 11) Scaling: 8T -> 16T

Comparison (6x16 TASK1):

| Size | Ours 8T | Ours 16T | Scale | OB 8T | OB 16T | OB Scale |
|---:|---:|---:|---:|---:|---:|---:|
| 256^3 | 247 | 278 | x1.13 | 222 | 317 | x1.43 |
| 512^3 | 384 | 538 | x1.40 | 252 | 425 | x1.69 |
| 1024^3 | 495 | 664 | x1.34 | 272 | 487 | x1.79 |
| 2048^3 | 459 | 766 | x1.67 | 558 | 517 | x0.93 |

Key fact: at 2048^3, OpenBLAS loses performance from 8T to 16T, while our kernel gains strongly.

---

# 12) Summary and Limitations

Completed:

- 4 micro-kernels (including ASM)
- BLIS 5-loop with packing
- TASK1 and TASK2 parallel paths
- Auto-tuning and adaptive benchmark
- 96/96 correctness tests passed

Current limitations and next steps:

1. TASK2: parallelize reduction and reduce partial-C footprint
2. Small matrices: add a lightweight fast path without full packing
3. Additional work: NUMA-aware pinning and scheduler tuning

---

# 13) Conclusion

Week 2 objective achieved:

- 8T: up to 182% of OpenBLAS
- 16T: up to 151% of OpenBLAS, peak 781 GF/s

Practical takeaway:

- The combination of micro-kernel design, cache blocking, and parallel decomposition is decisive
- Architecture-level choices matter more than isolated micro-optimizations
