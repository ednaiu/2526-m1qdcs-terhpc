import os
import sys
import subprocess
import re
import time

# Configuration
KERNEL = "saxpy"
MAX_ITERATIONS = 32
GOAL_GFLOPS = 40.0  # Performance goal
TEMPLATE_PATH = "blas1_template.c"
SRC_PATH = "../src/blas1.c"
BACKUP_PATH = "../src/blas1.c.bak"
BENCH_CMD = ["./../bin/bench_blas1", "--n", "10000000", "--kernel", KERNEL, "--mode", "avx2"]

def run_bench():
    """Compiles and runs the benchmark, returning median GFLOPS."""
    try:
        # Build
        subprocess.run(["make", "-C", "..", "bin/bench_blas1"], check=True, capture_output=True)
        # Run
        result = subprocess.run(BENCH_CMD, capture_output=True, text=True, check=True)
        # Parse output for GFLOPS
        # Expected format: saxpy,10000000,0.002,0.002,0.000,10.00,0.0,GFLOP/s
        # 4th comma-separated value is GFLOPS
        for line in result.stdout.splitlines():
            if line.startswith(f"{KERNEL},"):
                parts = line.split(",")
                if len(parts) >= 6:
                    return float(parts[5])
    except Exception as e:
        print(f"Error during benchmark: {e}")
    return 0.0

def generate_variant(unroll, prefetch, use_nt):
    """Generates a variant of blas1.c with the given parameters."""
    with open(TEMPLATE_PATH, "r") as f:
        content = f.read()
    
    content = content.replace("{{UNROLL_FACTOR}}", str(unroll))
    content = content.replace("{{PREFETCH_DIST}}", str(prefetch))
    
    # Generate Loop Body
    loop_body = ""
    store_instr = "_mm256_stream_ps" if use_nt else "_mm256_storeu_ps"
    for j in range(unroll):
        off = j * 8
        loop_body += f"        __m256 vx{j} = _mm256_loadu_ps(x + i + {off});\n"
        loop_body += f"        __m256 vy{j} = _mm256_loadu_ps(y + i + {off});\n"
        loop_body += f"        {store_instr}(y + i + {off}, _mm256_fmadd_ps(va, vx{j}, vy{j}));\n"
    
    # Generate Prefetch Logic
    prefetch_logic = ""
    if prefetch > 0:
        for j in range(unroll):
            off = j * 8
            prefetch_logic += f"        _mm_prefetch((const char *)(x + i + {off} + {prefetch}), _MM_HINT_T0);\n"
            prefetch_logic += f"        _mm_prefetch((const char *)(y + i + {off} + {prefetch}), _MM_HINT_T0);\n"

    content = content.replace("{{LOOP_BODY}}", loop_body)
    content = content.replace("{{PREFETCH_LOGIC}}", prefetch_logic)
    content = content.replace("{{SFENCE}}", "_mm_sfence();" if use_nt else "")

    with open(SRC_PATH, "w") as f:
        f.write(content)

def main():
    if not os.path.exists(BACKUP_PATH):
        subprocess.run(["cp", SRC_PATH, BACKUP_PATH], check=True)

    best_gflops = 0.0
    best_params = {}

    # Define search space
    unrolls = [1, 2, 4, 8]
    prefetches = [0, 32, 64, 128]
    nt_options = [False, True]

    iteration = 0
    for unroll in unrolls:
        for prefetch in prefetches:
            for nt in nt_options:
                if iteration >= MAX_ITERATIONS:
                    break
                
                print(f"Iteration {iteration}: Unroll={unroll}, Prefetch={prefetch}, NT={nt}")
                generate_variant(unroll, prefetch, nt)
                gflops = run_bench()
                print(f"Result: {gflops:.2f} GFLOPS")

                if gflops > best_gflops:
                    best_gflops = gflops
                    best_params = {"unroll": unroll, "prefetch": prefetch, "use_nt": nt}
                
                if best_gflops >= GOAL_GFLOPS:
                    print(f"Goal reached: {best_gflops:.2f} GFLOPS")
                    break
                
                iteration += 1
            if best_gflops >= GOAL_GFLOPS or iteration >= MAX_ITERATIONS: break
        if best_gflops >= GOAL_GFLOPS or iteration >= MAX_ITERATIONS: break

    print("\n" + "="*20)
    print(f"Optimization Finished")
    print(f"Best GFLOPS: {best_gflops:.2f}")
    print(f"Best Params: {best_params}")
    print("="*20)

    # Apply best result
    if best_params:
        generate_variant(**best_params)
        print("Best variant applied to src/blas1.c")
    else:
        print("No successful optimization iteration found. Restoring backup.")
        subprocess.run(["cp", BACKUP_PATH, SRC_PATH], check=True)

if __name__ == "__main__":
    main()
