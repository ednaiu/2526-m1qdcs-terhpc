# Week 4 Comprehensive Performance Report

This report details the implementation, optimization steps, and exhaustive performance analysis of the updates introduced in Week 4. The work centers on two primary objectives:
1.  **BLAS 1 Kernels Development**: Building foundational vector operations from scratch, incrementally applying optimizations (SIMD, latency hiding, multi-threading), and benchmarking against OpenBLAS.
2.  **SGEMM Parallel Tasking Analysis**: Investigating the architectural differences and performance trade-offs between Loop-based (`omp for`) and Task-based (`omp task`) parallel scheduling models for matrix multiplication.

All benchmarks are evaluated using an adaptive timing mechanism. Each configuration is executed multiple times (minimum 5, up to 64) until the standard deviation falls below 2%, ensuring statistical accuracy and mitigating system noise. The reported value is the median of these runs.

---

## Part 1: BLAS 1 Library Optimization

BLAS Level 1 operations are the fundamental tools of linear algebra—vector scaling, dot products, additions, etc. While they are computationally simple (often just $O(N)$ operations), they are highly constrained by memory bandwidth.

We implemented 9 kernels: `sscal`, `scopy`, `sswap`, `saxpy`, `sdot`, `snrm2`, `sasum`, `isamax`, and `srot`. To accurately measure the impact of each optimization technique, we developed the kernels in three progressive stages.

### Concept 1: The Scalar Baseline (The "Naive" Approach)

**Explanation:**
The scalar baseline represents the most straightforward way to write the algorithm in C using a simple `for` loop, processing one element at a time. It relies entirely on the compiler and the CPU's default mechanisms to handle memory and computation.

**Impact & Results:**
As expected, processing one element per clock cycle fails to fully utilize the CPU's capabilities. For memory-bound operations, it slowly fetches data. For compute-bound operations (like `snrm2` or `sdot`), it leaves the arithmetic logic units (ALUs) starved.

*Sample Results (N = 1,048,576)*
| Kernel | Metric | Scalar Result | OpenBLAS Result | % of OpenBLAS |
| :--- | :--- | :--- | :--- | :--- |
| `sscal` | GB/s | 19.46 | 18.12 | 107% |
| `scopy` | GB/s | 21.78 | 17.95 | 121% |
| `saxpy` | GFLOP/s | 5.54 | 41.72 | **13%** |
| `sdot` | GFLOP/s | 1.68 | 7.51 | **22%** |
| `snrm2` | GFLOP/s | 1.70 | 5.55 | **30%** |

*Observation*: While simple memory moves (`scopy`, `sscal`) performed decently purely due to the CPU's hardware prefetcher, compute-involved operations (`saxpy`, `sdot`) were drastically slower than the highly tuned OpenBLAS.

### Concept 2: SIMD Vectorization (AVX2)

**Explanation:**
Modern CPUs feature wide execution lanes known as SIMD (Single Instruction, Multiple Data). Using AVX2 intrinsics, we manually rewrote the kernels to load 8 floating-point numbers into a 256-bit register, perform the arithmetic (e.g., 8 multiplications at once using FMA), and store 8 results back to memory in a single instruction sequence. We essentially upgraded from carrying items one-by-one by hand to using a large wheelbarrow.

**Impact & Results:**
Vectorization immediately removed computation bottlenecks for a single thread.

*Sample Results (N = 1,048,576)*
| Kernel | Metric | Scalar | AVX2 | Speedup over Scalar |
| :--- | :--- | :--- | :--- | :--- |
| `sscal` | GB/s | 19.46 | 19.78 | 1.01x |
| `saxpy` | GFLOP/s | 5.54 | 5.56 | 1.00x |
| `sdot` | GFLOP/s | 1.68 | 7.32 | **4.35x** |
| `snrm2` | GFLOP/s | 1.70 | 14.99 | **8.81x** |
| `sasum` | GB/s | 3.39 | 30.01 | **8.85x** |

*Observation*: Reductions (`sdot`, `snrm2`, `sasum`) saw massive improvements (up to 8.8x) because the CPU could aggressively accumulate 8 partial sums simultaneously. Operations limited entirely by how fast RAM can be read/written (`sscal`, `saxpy`) showed no improvement because the single thread had already saturated its allocated memory bandwidth.

### Concept 3: Multi-threading & Latency Hiding (AVX2 + OpenMP)

**Explanation:**
To push past the single-core memory bandwidth limit and instruction latency, we combined two advanced techniques:
1.  **OpenMP Parallelism**: We divided the giant vectors into chunks and assigned them to multiple CPU cores (8 threads).
2.  **Latency Hiding (Independent Accumulators)**: Even within AVX2, an instruction like FMA takes several clock cycles. We unrolled the micro-loops to use 4 or 8 independent accumulator registers. The CPU can process the second set of 8 numbers while the first set is still physically traveling through the floating-point hardware unit.

**Impact & Results:**
This combination unlocks the peak performance of the machine, saturating the entire CPU socket's memory bus and arithmetic units.

*Final Comparative Results (N = 1,048,576, 8 Threads)*
| Kernel | Metric | Scalar | AVX2 | AVX2+OMP | OpenBLAS | % of OpenBLAS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sscal` | GB/s | 19.46 | 19.78 | **134.40** | 18.12 | **741%** |
| `scopy` | GB/s | 21.78 | 22.10 | **156.40** | 17.95 | **871%** |
| `saxpy` | GFLOP/s | 5.54 | 5.56 | **42.63** | 41.72 | **102%** |
| `sdot` | GFLOP/s | 1.68 | 7.32 | **54.60** | 7.51 | **727%** |
| `snrm2` | GFLOP/s | 1.70 | 14.99 | **91.98** | 5.55 | **1658%** |

*Observation*: By leveraging all 8 threads and hiding instruction latency, our implementation obliterates OpenBLAS in almost every category, especially in reductions (`sdot`, `snrm2`), achieving up to 16x the performance of the established standard library. OpenBLAS BLAS1 functions are often surprisingly unoptimized for multi-threading on large arrays; our explicit OpenMP distribution proves vastly superior.

---

## Part 2: SGEMM Parallel Scheduling (Loops vs. Tasks)

In Week 3, we successfully distributed SGEMM across multiple cores. In Week 4, we undertook a comparative analysis of **how** that work is distributed, comparing traditional Loop scheduling with modern Task scheduling.

### Concept 1: Loop-Based Parallelism (`omp for collapse`)

**Explanation:**
Loop-based parallelism statically or dynamically slices a nested set of `for` loops and assigns exact ranges of iterations to specific threads. 
*Analogy*: An established assembly line where every worker knows exactly which widgets (tiles of matrix `C`) they are responsible for building before the shift even begins.

**Impact:**
Because the plan is made upfront and the iteration space is perfectly regular (a 2D grid), the overhead is virtually zero. 

### Concept 2: Task-Based Parallelism (`omp task` / `omp taskloop`)

**Explanation:**
Task-based parallelism treats every iteration (every tile of matrix `C`) as an independent "Task". A single master thread spawns these tasks, packing all the required variables (`firstprivate`) into a descriptor block in memory, and pushes them into a queue. Any idle thread can pop a task and execute it.
*Analogy*: A ticket-based kitchen line. When an order (tile) comes in, a ticket is printed. Any available chef can grab the next ticket and cook it. It is incredibly flexible for unbalanced workloads but requires time to print and read the tickets.

### Comparative Results (GFLOP/s)

We benchmarked the `6x16` micro-kernel using both paradigms in 2D parallel configuration (TASK1) using 8 threads. 

| Matrix Size | TASK1 [LOOP] | TASK1 [TASK] | OpenBLAS | Loop vs Task Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **64** | 41.38 | 41.82 | 50.18 | 1.0x (Noise bounds) |
| **128** | **85.16** | 6.09 | 141.33 | **Task is 14.0x Slower** |
| **256** | **208.03** | 26.06 | 283.86 | **Task is 8.0x Slower** |
| **512** | **341.91** | 204.30 | 400.22 | **Task is 1.6x Slower** |
| **1024** | **420.54** | 378.00 | 463.76 | **Task is 1.1x Slower** |
| **2048** | **452.23** | 435.48 | 505.12 | **Task is 1.03x Slower** |

### Analysis of the Crossover Point

1.  **The Small Matrix Disaster (N < 256):** 
    For a 128x128 matrix, the work required to multiply a $6 \times 16$ tile is infinitesimal (measured in instructions). The time it takes OpenMP to allocate a block of memory, capture the local variables, push the task to a queue, and have a worker thread pop it is orders of magnitude larger than the math itself. **Task overhead dominates execution time.**
2.  **The Convergence (N $\ge$ 1024):**
    Matrix multiplication scales as $O(N^3)$ for computation, but only $O(N^2)$ for the number of grid tiles (tasks). Therefore, as the matrix grows, the time spent actually doing math inside the micro-kernel dwarfs the initial time spent printing the "task tickets". By $N=2048$, the overhead difference is merely ~3.7%.

### Conclusion
For regular, dense, predictable grid workloads like Matrix Multiplication (SGEMM), **Loop-based scheduling (`omp for`) is overwhelmingly superior**. Task-based scheduling, while powerful for irregular applications (like tree traversals or graphs), introduces unacceptable overhead for small matrix operations. By relying on `omp for collapse(2) schedule(dynamic)`, our library guarantees both near-zero scheduling overhead and perfect load balancing across all cores.
