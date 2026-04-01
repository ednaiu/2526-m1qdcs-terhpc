/*
 * NOTE: _GNU_SOURCE exposes sched_getcpu() on Linux.
 */
#define _GNU_SOURCE
/*
 * hello_omp.c  —  OpenMP Hello World
 *
 * Compile:  gcc -O2 -fopenmp -o hello_omp hello_omp.c
 * Run:      OMP_NUM_THREADS=8 ./hello_omp
 */

#include <stdio.h>
#include <omp.h>
#include <sched.h>

int main(void)
{
    printf("OpenMP version: %d\n", _OPENMP);

#pragma omp parallel
    {
        int tid     = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        /* Each thread prints its own greeting (order may vary). */
#pragma omp critical
        printf("Hello from thread %d / %d  (running on core %d)\n",
               tid, nthreads, sched_getcpu());
    }

    return 0;
}
