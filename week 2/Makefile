# Makefile for Week 2: SGEMM with multiple kernels and autotuning

# Compiler and flags
CC        := gcc
CFLAGS    := -O3 -march=native -fPIC -Wall -Wextra
CFLAGS    += -fopenmp -mavx -mavx2 -mfma
LDFLAGS   := -fopenmp -lm

# OpenBLAS (for comparison binary)
OPENBLAS_CFLAGS  := $(shell pkg-config --cflags openblas 2>/dev/null)
OPENBLAS_LDFLAGS := $(shell pkg-config --libs openblas 2>/dev/null || echo "-lopenblas")

# Directories
SRC_DIR   := src
BENCH_DIR := bench
TEST_DIR  := tests
INC_DIR   := include
BIN_DIR   := bin
AUTO_DIR  := autotuning
RES_DIR   := results

# Source files
KERN_SRCS := $(wildcard $(SRC_DIR)/kernels/*.h)
SGEMM_SRC := $(SRC_DIR)/sgemm.c
BENCH_SRC := $(BENCH_DIR)/bench_sgemm.c
CMP_SRC   := $(BENCH_DIR)/compare_direct_openblas.c
TEST_SRC  := $(TEST_DIR)/test_sgemm.c

# Object files
SGEMM_OBJ := $(BIN_DIR)/sgemm.o
BENCH_OBJ := $(BIN_DIR)/bench_sgemm.o

# Binaries
BENCH_BIN := $(BIN_DIR)/bench_sgemm
TEST_BIN  := $(BIN_DIR)/test_sgemm
CMP_BIN   := $(BIN_DIR)/compare_direct_openblas
SIMPLE_BIN := $(BIN_DIR)/simple_test

# Targets
.PHONY: all clean bench test compare simple-test \
        run-bench run-test run-simple run-compare run-all \
        help check-compiler

# Default target
all: $(BENCH_BIN) $(SIMPLE_BIN)

# Help target
help:
	@echo "Week 2 SGEMM Build System"
	@echo "=========================="
	@echo ""
	@echo "Build targets:"
	@echo "  make all           — Build benchmark + simple test"
	@echo "  make bench         — Build benchmark binary"
	@echo "  make test          — Build full test suite"
	@echo "  make simple-test   — Build lightweight correctness tests"
	@echo "  make compare       — Build direct OpenBLAS comparison binary"
	@echo "  make clean         — Remove all build artifacts"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-bench     — Build and run benchmark suite (CSV)"
	@echo "  make run-simple    — Build and run lightweight tests"
	@echo "  make run-compare   — Build and run OpenBLAS comparison"
	@echo "  make run-all       — Run all tests and benchmarks"
	@echo ""
	@echo "Benchmark CLI (single mode):"
	@echo "  ./bin/bench_sgemm --M 1024 --kernel 6x16 --threads 4"
	@echo "  ./bin/bench_sgemm --M 512 --MC 120 --KC 512 --NC 4096"
	@echo ""
	@echo "Python autotuning:"
	@echo "  python3 autotuning/benchmark_runner.py --M 1024 -v"
	@echo "  python3 autotuning/gradient_descent.py --M 1024 --threads 4 -v"
	@echo "  python3 autotuning/bayesian_opt.py --M 1024 --threads 4 -v"
	@echo "  python3 autotuning/simulated_annealing.py --M 1024 -v"
	@echo "  python3 autotuning/random_search.py --M 1024 -v"
	@echo "  python3 autotuning/compare_algorithms.py --M 512 --budget 30"
	@echo "  python3 autotuning/kernel_comparison.py"
	@echo "  python3 autotuning/tune_3d_parallelism.py"
	@echo "  python3 autotuning/compare_with_openblas.py"

# Create directories
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(RES_DIR):
	@mkdir -p $(RES_DIR)

# Compile SGEMM library
$(SGEMM_OBJ): $(SGEMM_SRC) $(KERN_SRCS) $(INC_DIR)/sgemm.h | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $(SGEMM_SRC) -o $@

# Build benchmark — plain version (CLI autotuning, JSON output)
$(BENCH_OBJ): $(BENCH_SRC) $(INC_DIR)/sgemm.h | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $(BENCH_SRC) -o $@

$(BENCH_BIN): $(SGEMM_OBJ) $(BENCH_OBJ) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build benchmark with OpenBLAS table mode (enables bench_table())
$(BIN_DIR)/bench_sgemm_ob.o: $(BENCH_SRC) $(INC_DIR)/sgemm.h | $(BIN_DIR)
	$(CC) $(CFLAGS) $(OPENBLAS_CFLAGS) -DWITH_OPENBLAS -I$(INC_DIR) -c $(BENCH_SRC) -o $@

$(BIN_DIR)/bench_sgemm_ob: $(SGEMM_OBJ) $(BIN_DIR)/bench_sgemm_ob.o | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(OPENBLAS_LDFLAGS)

# Build direct OpenBLAS comparison
$(CMP_BIN): $(SGEMM_OBJ) $(CMP_SRC) $(INC_DIR)/sgemm.h | $(BIN_DIR)
	$(CC) $(CFLAGS) $(OPENBLAS_CFLAGS) -I$(INC_DIR) -o $@ $(CMP_SRC) $(SGEMM_OBJ) $(LDFLAGS) $(OPENBLAS_LDFLAGS)

# Build test suite
$(BIN_DIR)/test_sgemm.o: $(TEST_SRC) $(INC_DIR)/sgemm.h | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $(TEST_SRC) -o $@

$(TEST_BIN): $(SGEMM_OBJ) $(BIN_DIR)/test_sgemm.o | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build simple test
$(BIN_DIR)/simple_test.o: $(TEST_DIR)/simple_test.c $(INC_DIR)/sgemm.h | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $(TEST_DIR)/simple_test.c -o $@

$(SIMPLE_BIN): $(SGEMM_OBJ) $(BIN_DIR)/simple_test.o | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Convenience targets
bench: $(BENCH_BIN)
bench-ob: $(BIN_DIR)/bench_sgemm_ob
test: $(TEST_BIN)
compare: $(CMP_BIN)
simple-test: $(SIMPLE_BIN)

# Run targets
run-bench: $(BIN_DIR)/bench_sgemm_ob | $(RES_DIR)
	@echo "Running SGEMM vs OpenBLAS table..."
	OMP_NUM_THREADS=$$(nproc) OPENBLAS_NUM_THREADS=$$(nproc) \
	    $(BIN_DIR)/bench_sgemm_ob | tee $(RES_DIR)/bench_$$(date +%Y%m%d_%H%M%S).txt

run-simple: $(SIMPLE_BIN)
	@echo "Running lightweight SGEMM tests..."
	@./$(SIMPLE_BIN)

run-test: $(TEST_BIN)
	@echo "Running SGEMM correctness tests..."
	@./$(TEST_BIN)

run-compare: $(CMP_BIN)
	@echo "Running direct OpenBLAS comparison..."
	@./$(CMP_BIN) 5

run-all: run-simple run-bench
	@echo "All tests and benchmarks completed."

# Clean build artifacts
clean:
	@rm -rf $(BIN_DIR)
	@echo "Cleaned build artifacts."

# Compiler info
check-compiler:
	@echo "Compiler: $(CC)"
	@$(CC) --version
	@echo ""
	@echo "CFLAGS: $(CFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"
	@echo "OPENBLAS: $(OPENBLAS_LDFLAGS)"
