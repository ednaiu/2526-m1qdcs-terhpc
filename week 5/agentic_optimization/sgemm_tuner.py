import subprocess
import json
import os

# Configuration
M, N, K = 1024, 1024, 1024
THREADS = 16
MR, NR = 6, 16

# Search space
MC_LIST = [60, 96, 120, 150, 192]
KC_LIST = [256, 384, 512, 768]
NC_LIST = [2048, 4096, 8192]

best_gflops = 0
best_config = None

print(f"Starting SGEMM Autotuner (M=N=K={M}, Threads={THREADS})")
print(f"{'MC':>4} {'KC':>4} {'NC':>5} | {'GFLOPS':>10}")
print("-" * 30)

for nc in NC_LIST:
    for kc in KC_LIST:
        for mc in MC_LIST:
            # Ensure MC is multiple of MR
            mc_adj = (mc // MR) * MR
            
            cmd = [
                "../bin/bench_sgemm",
                "--M", str(M),
                "--MC", str(mc_adj),
                "--KC", str(kc),
                "--NC", str(nc),
                "--threads", str(THREADS),
                "--min-runs", "3",
                "--max-runs", "5",
                "--target-stddev", "3.0"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                data = json.loads(result.stdout)
                gf = data["gflops_median"]
                
                print(f"{mc_adj:4d} {kc:4d} {nc:5d} | {gf:10.2f}")
                
                if gf > best_gflops:
                    best_gflops = gf
                    best_config = (mc_adj, kc, nc)
            except Exception as e:
                print(f"Error at MC={mc_adj}, KC={kc}, NC={nc}: {e}")

print("-" * 30)
if best_config:
    mc, kc, nc = best_config
    print(f"Best Configuration: MC={mc}, KC={kc}, NC={nc}")
    print(f"Performance: {best_gflops:.2f} GFLOPS")
    
    # Update sgemm.h with the best defaults
    print("\nTo apply these results, update SGEMM_DEFAULT_CONFIG in include/sgemm.h")
