#!/bin/bash
echo "| Threads | Loop N=1024 | Task N=1024 | Loop N=2048 | Task N=2048 |" > thread_bench.md
echo "|---|---|---|---|---|" >> thread_bench.md
for t in 1 2 4 8 12 16 20; do
    export OMP_NUM_THREADS=$t
    l1=$(./bin/bench_sgemm --M 1024 --parallel 2D --sched loop --threads $t | grep Performance | awk '{print $2}')
    t1=$(./bin/bench_sgemm --M 1024 --parallel 2D --sched task --threads $t | grep Performance | awk '{print $2}')
    l2=$(./bin/bench_sgemm --M 2048 --parallel 2D --sched loop --threads $t | grep Performance | awk '{print $2}')
    t2=$(./bin/bench_sgemm --M 2048 --parallel 2D --sched task --threads $t | grep Performance | awk '{print $2}')
    echo "| $t | $l1 GF | $t1 GF | $l2 GF | $t2 GF |" >> thread_bench.md
done
cat thread_bench.md
