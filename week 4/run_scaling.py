import sys, json, subprocess, os

threads = [1, 2, 4, 8, 12, 16, 20]
sizes = [1024, 2048]

cmd_base = "./bin/bench_sgemm --M {M} --parallel 2D --sched {s} --threads {t}"

print("| Threads | Loop N=1024 | Task N=1024 | Loop N=2048 | Task N=2048 |")
print("|---|---|---|---|---|")

# ensure OMP_NUM_THREADS doesn't interfere if passing --threads, but we can set it anyway
for t in threads:
    os.environ['OMP_NUM_THREADS'] = str(t)
    row = [str(t)]
    for M in sizes:
        for sched in ['loop', 'task']:
            cmd = cmd_base.format(M=M, s=sched, t=t)
            res = subprocess.run(cmd.split(), capture_output=True, text=True)
            try:
                data = json.loads(res.stdout)
                gf = f"{data['gflops_median']:.2f}"
            except:
                gf = "N/A"
            row.append(gf)
    print(f"| {' | '.join(row)} |")
