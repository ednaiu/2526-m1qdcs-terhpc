#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <omp.h>
#include <math.h>
#include <cblas.h>

#include "../include/sgemm.h"

extern void openblas_set_num_threads(int num_threads);

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static int cmp_double(const void *a, const void *b) {
    double da=*(const double *)a, db=*(const double *)b;
    return (da>db)-(da<db);
}
static double median(double *arr, int n) {
    double tmp[64]; if(n>64)n=64; memcpy(tmp, arr, n*sizeof(double));
    qsort(tmp, n, sizeof(double), cmp_double);
    return (n%2==1)?tmp[n/2]:0.5*(tmp[n/2-1]+tmp[n/2]);
}
static double stddev_d(double *a, int n, double mean) {
    double s=0; for(int i=0;i<n;i++) s+=(a[i]-mean)*(a[i]-mean);
    return sqrt(s/n);
}

static void pin_to_first_n_cpus(int n) {
    cpu_set_t set; CPU_ZERO(&set);
    for(int i=0;i<n;++i) CPU_SET(i, &set);
    sched_setaffinity(0, sizeof(set), &set);
}
static void fill_random(float *x, size_t n) {
    for(size_t i=0;i<n;++i) x[i] = (float)rand()/(float)RAND_MAX - 0.5f;
}
static double run_openblas_once(int M, int N, int K, const float *A, const float *B, float *C) {
    double t1 = now_sec();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    return now_sec() - t1;
}
static double run_week2_once(int M, int N, int K, const float *A, const float *B, float *C, int threads) {
    sgemm_config_t cfg = SGEMM_DEFAULT_CONFIG;
    cfg.kernel = KERNEL_6x16; cfg.parallel_mode = PARALLEL_2D; cfg.nb_threads = threads;
    double t1 = now_sec();
    sgemm_ex(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N, &cfg);
    return now_sec() - t1;
}

static void bench_single_openblas(int M, int N, int K, int threads, int min_runs, int max_runs, double target_cv_pct) {
    size_t asz=(size_t)M*K, bsz=(size_t)K*N, csz=(size_t)M*N;
    float *A=malloc(asz*4), *B=malloc(bsz*4), *C=malloc(csz*4);
    fill_random(A, asz); fill_random(B, bsz); memset(C, 0, csz*4);

    pin_to_first_n_cpus(threads);
    openblas_set_num_threads(threads); omp_set_num_threads(threads);

    run_openblas_once(M, N, K, A, B, C); // warmup

    double times[64] = {0}; int n = 0;
    for(n=0; n<max_runs; n++) {
        memset(C, 0, csz*4);
        times[n] = run_openblas_once(M, N, K, A, B, C);
        if(n+1 >= min_runs) {
            double mean=0; for(int i=0;i<=n;i++) mean+=times[i];
            mean/=(n+1);
            double sd=stddev_d(times, n+1, mean);
            if(mean>0 && (sd/mean*100.0)<target_cv_pct) { n++; break; }
        }
    }
    double mean=0, mn=times[0], mx=times[0];
    for(int i=0; i<n; i++){
        mean+=times[i]; if(times[i]<mn) mn=times[i]; if(times[i]>mx) mx=times[i];
    }
    mean/=n; double med=median(times, n); double sd=stddev_d(times, n, mean);
    double cv = mean>0 ? sd/mean*100.0 : 0.0;
    
    #define GFLOPS(t) ((t)>0 ? 2.0*M*N*K/(t)/1e9 : 0.0)
    printf("{\n  \"runs\": %d,\n  \"median_s\": %.9f,\n  \"mean_s\": %.9f,\n", n, med, mean);
    printf("  \"min_s\": %.9f,\n  \"max_s\": %.9f,\n  \"stddev_s\": %.9f,\n", mn, mx, sd);
    printf("  \"stddev_pct\": %.4f,\n  \"gflops_median\": %.4f,\n", cv, GFLOPS(med));
    printf("  \"gflops_mean\": %.4f,\n  \"gflops_min\": %.4f,\n  \"gflops_max\": %.4f,\n", GFLOPS(mean), GFLOPS(mx), GFLOPS(mn));
    printf("  \"all_times\": [");
    for(int i=0;i<n;i++) printf("%s%.9f", i?",":"", times[i]);
    printf("]\n}\n");
    free(A); free(B); free(C);
}

static void benchmark_case(int M, int N, int K, int threads, int runs, FILE *csv) {
    size_t asz=M*K, bsz=K*N, csz=M*N;
    float *A=malloc(asz*4), *B=malloc(bsz*4), *C_ob=malloc(csz*4), *C_w2=malloc(csz*4);
    fill_random(A,asz); fill_random(B,bsz);
    pin_to_first_n_cpus(threads); openblas_set_num_threads(threads); omp_set_num_threads(threads);

    run_openblas_once(M,N,K,A,B,C_ob); run_week2_once(M,N,K,A,B,C_w2,threads);

    double flops=2.0*M*N*K, *ob_gf=malloc(runs*8), *w2_gf=malloc(runs*8);
    for(int i=0;i<runs;++i){
        memset(C_ob,0,csz*4); memset(C_w2,0,csz*4);
        ob_gf[i]=flops/run_openblas_once(M,N,K,A,B,C_ob)/1e9;
        w2_gf[i]=flops/run_week2_once(M,N,K,A,B,C_w2,threads)/1e9;
    }
    double ob_med=median(ob_gf,runs), w2_med=median(w2_gf,runs), ratio=ob_med>0?w2_med/ob_med:0;
    printf("%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%s\n", M,N,K,threads,runs,ob_med,w2_med,ratio,ratio>=1.0?"Week2":"OpenBLAS");
    if(csv) fprintf(csv,"%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%s\n", M,N,K,threads,runs,ob_med,w2_med,ratio,ratio>=1.0?"Week2":"OpenBLAS");
    free(ob_gf); free(w2_gf); free(A); free(B); free(C_ob); free(C_w2);
}

static int get_int_arg(int argc, char **argv, const char *flag, int def) {
    for(int i=1;i<argc-1;i++) if(!strcmp(argv[i],flag)) return atoi(argv[i+1]);
    return def;
}
static double get_dbl_arg(int argc, char **argv, const char *flag, double def) {
    for(int i=1;i<argc-1;i++) if(!strcmp(argv[i],flag)) return atof(argv[i+1]);
    return def;
}
static int has_flag(int argc, char **argv, const char *flag) {
    for(int i=1;i<argc;i++) if(!strcmp(argv[i],flag)) return 1;
    return 0;
}

int main(int argc, char **argv) {
    srand(12345);
    if (has_flag(argc, argv, "--M")) {
        int M = get_int_arg(argc, argv, "--M", 1024);
        int N = get_int_arg(argc, argv, "--N", M);
        int K = get_int_arg(argc, argv, "--K", M);
        int threads = get_int_arg(argc, argv, "--threads", 1);
        int min_runs = get_int_arg(argc, argv, "--min-runs", 3);
        int max_runs = get_int_arg(argc, argv, "--max-runs", 15);
        double target_cv = get_dbl_arg(argc, argv, "--target-stddev", 2.0);
        bench_single_openblas(M, N, K, threads, min_runs, max_runs, target_cv);
        return 0;
    }

    int runs=5; if(argc>=2 && !has_flag(argc,argv,"--M")) { runs=atoi(argv[1]); if(runs<=0) runs=5; }
    int sizes[]={256,512,1024}, threads_list[]={8,16};
    FILE *csv=fopen("results/direct_openblas_vs_week2.csv","w");
    printf("M,N,K,threads,runs,openblas_median_gflops,week2_median_gflops,ratio_week2_over_openblas,winner\n");
    if(csv) fprintf(csv,"M,N,K,threads,runs,openblas_median_gflops,week2_median_gflops,ratio_week2_over_openblas,winner\n");
    for(size_t si=0;si<3;++si) for(size_t ti=0;ti<2;++ti) benchmark_case(sizes[si],sizes[si],sizes[si],threads_list[ti],runs,csv);
    if(csv) fclose(csv);
    return 0;
}
