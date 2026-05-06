# ==========================================================================
# libmyblas — Fortran-compatible single-precision BLAS 1/2/3 library
# ==========================================================================
#
# Build:    make          (or: make all)
# Clean:    make clean
# Outputs:  lib/libmyblas.a      — static archive
#           lib/libmyblas.so     — shared object (soname libmyblas.so.1)
#
# Linking against the static library:
#   gcc mytest.c -L./lib -lmyblas -fopenmp -lm -o mytest
#
# Linking against the shared library:
#   gcc mytest.c -L./lib -Wl,-rpath,'$$ORIGIN/lib' -lmyblas -fopenmp -lm -o mytest
#
# Replacing OpenBLAS in an existing link line — just swap the library:
#   was:  ... -lopenblas
#   now:  ... -L./lib -lmyblas -fopenmp -lm
#
# --------------------------------------------------------------------------
# Sources: week 5/src/   (final, best implementation of all BLAS levels)
#
#   sgemm.c         SGEMM with BLIS 5-loop, AVX2 micro-kernels (6×16 default)
#   blas1.c         9 kernels: sscal scopy sswap saxpy sdot snrm2 sasum isamax srot
#   blas2.c         7 kernels: sgemv sger ssymv strmv strsv ssyr ssyr2
#   blas3.c         3 kernels: ssyrk strsm strmm
#   blas_fortran.c  sgemm_() Fortran entry point (col-major ↔ row-major bridge)
#
# Fortran symbols exported (trailing-underscore, all args by pointer):
#   BLAS-1: sscal_ scopy_ sswap_ saxpy_ sdot_ snrm2_ sasum_ isamax_ srot_
#   BLAS-2: sgemv_ sger_ ssymv_ strmv_ strsv_ ssyr_ ssyr2_
#   BLAS-3: sgemm_ ssyrk_ strsm_ strmm_
#
# All symbols are ABI-compatible with gfortran (Linux x86-64, default
# -fdefault-integer-4).  The same test executable can be linked against
# this library or against OpenBLAS without modifying any source code.
# ==========================================================================

CC      := gcc
CFLAGS  := -O3 -march=native -fPIC -fopenmp -mavx2 -mfma -Wall -Wextra
LDLIBS  := -fopenmp -lm

LIBDIR  := lib
OBJDIR  := $(LIBDIR)/obj

STATIC  := $(LIBDIR)/libmyblas.a
SHARED  := $(LIBDIR)/libmyblas.so

# All object files
OBJS :=	$(OBJDIR)/sgemm.o        \
	$(OBJDIR)/blas1.o        \
	$(OBJDIR)/blas2.o        \
	$(OBJDIR)/blas3.o        \
	$(OBJDIR)/blas_fortran.o

# Include path (quoted in recipes because of the space in "week 5")
INC := week 5/include

.PHONY: all clean help

all: $(STATIC) $(SHARED)
	@echo ""
	@echo "  $(STATIC)"
	@echo "  $(SHARED)"
	@echo ""
	@echo "Link (static):  gcc test.c -L./lib -lmyblas -fopenmp -lm"
	@echo "Link (shared):  gcc test.c -L./lib -Wl,-rpath,\$$ORIGIN/lib -lmyblas -fopenmp -lm"

# ---- directories ---------------------------------------------------------
$(OBJDIR):
	mkdir -p "$(OBJDIR)"

# ---- compilation ---------------------------------------------------------
# Source paths contain a space ("week 5/src/...").
# We avoid GNU Make's space-as-separator problem by hardcoding the quoted
# shell paths directly in every recipe and using order-only prerequisites
# (| $(OBJDIR)) instead of file-based prerequisites for the sources.
# If you change a source file, run `make clean && make` to force a rebuild.

$(OBJDIR)/sgemm.o: | $(OBJDIR)
	$(CC) $(CFLAGS) -I"$(INC)" -c "week 5/src/sgemm.c" -o $@

$(OBJDIR)/blas1.o: | $(OBJDIR)
	$(CC) $(CFLAGS) -I"$(INC)" -c "week 5/src/blas1.c" -o $@

$(OBJDIR)/blas2.o: | $(OBJDIR)
	$(CC) $(CFLAGS) -I"$(INC)" -c "week 5/src/blas2.c" -o $@

$(OBJDIR)/blas3.o: | $(OBJDIR)
	$(CC) $(CFLAGS) -I"$(INC)" -c "week 5/src/blas3.c" -o $@

$(OBJDIR)/blas_fortran.o: | $(OBJDIR)
	$(CC) $(CFLAGS) -I"$(INC)" -c "week 5/src/blas_fortran.c" -o $@

# ---- static library ------------------------------------------------------
$(STATIC): $(OBJS)
	ar rcs $@ $^

# ---- shared library ------------------------------------------------------
# soname = libmyblas.so.1  (major version only, consistent with OpenBLAS convention)
$(SHARED): $(OBJS)
	$(CC) -shared -Wl,-soname,libmyblas.so.1 -o $@ $^ $(LDLIBS)

# ---- housekeeping --------------------------------------------------------
clean:
	rm -rf $(LIBDIR)

help:
	@echo "Targets:"
	@echo "  all    — build lib/libmyblas.a and lib/libmyblas.so  (default)"
	@echo "  clean  — remove the lib/ directory"
	@echo "  help   — this message"
	@echo ""
	@echo "Fortran BLAS-1 symbols:"
	@echo "  sscal_  scopy_  sswap_  saxpy_  sdot_  snrm2_  sasum_  isamax_  srot_"
	@echo ""
	@echo "Fortran BLAS-2 symbols:"
	@echo "  sgemv_  sger_  ssymv_  strmv_  strsv_  ssyr_  ssyr2_"
	@echo ""
	@echo "Fortran BLAS-3 symbols:"
	@echo "  sgemm_  ssyrk_  strsm_  strmm_"
