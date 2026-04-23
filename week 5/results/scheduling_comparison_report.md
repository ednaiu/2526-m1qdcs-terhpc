# SGEMM Scheduling Comparison: Loops vs Tasks

| Size | OpenBLAS | TASK1 [LOOP] | TASK1 [TASK] | TASK2 [LOOP] | TASK2 [TASK] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **128** | 141.3 | 85.2 (60%) | 6.1 (4%) | 4.2 (3%) | 0.9 (1%) |
| **512** | 400.2 | 341.9 (85%) | 204.3 (51%) | 22.2 (6%) | 6.3 (2%) |
| **1024** | 463.8 | 420.5 (91%) | 378.0 (82%) | 60.6 (13%) | 24.9 (5%) |
| **2048** | 505.1 | 452.2 (90%) | 435.5 (86%) | 140.7 (28%) | 127.6 (25%) |

## Analysis
Task-based scheduling is significantly slower for small problem sizes due to thread management and variable capture overhead. For large matrices, the difference becomes negligible as the $O(N^3)$ computational work starts to dominate the $O(N^2)$ task overhead. For regular applications like GEMM, `#pragma omp for collapse` is generally preferred.
