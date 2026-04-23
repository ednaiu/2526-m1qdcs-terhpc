# 2526-m1qdcs-terhpc

In this TER project, we will develop, using AI-driven HPC software development, a fully functional and optimized BLAS library that rivals the performance of OpenBLAS; a hand-tuned optimized multithreaded BLAS/LAPACK implementation.

# Organization

* First of all, please make sure that your name and e-mail appears in the list of registered students (if not please add it): https://cirrus.universite-paris-saclay.fr/s/E55T4HoNWnZZx23

* We will be using Google Antigravity IDE as the AI assistant tool (https://antigravity.google). With your university account, you can get a one-year free Google Gemini student membership (https://gemini.google/students/); please go ahead and activate your Gemini account, which we will use as the AI backend of the Antigravity IDE.

* I will give you access to a 20-core Intel CPU workstation. For this, I will need you to put your SSH key in the registration sheet. Please put your **public ssh key (ending with .pub)** on ssh key column in the spreadsheet above. There you will find the ssh command to connect to your account on the remote machine.

* At the beginning of each week's session, I would like to have each group a mini talk/presentation of 5-6 minutes, sharing their experience regarding their work in the previous week. This will be an informal exchange; you do not need a polished talk or slides. Talking points: what worked, what did not work, challenges faced, how managed to overcome, etc. This will also give you little bouts of storyline for your final presentation and report.

* After the presentations, I will talk with each group and discuss the week's work, inspect the code, prompts, results, and give guidance for further development.

## Week 1: Getting started with Antigravity and tiled matrix multiplication SGEMM (13/03)

Goals for the week

* Installing and setting up antigravity

* Setting up connection to the remote workstation through Antigravity

* Launching agents and creating a OpenMP Hello World program that displays thread id and number of threads.

* 8x8 matrix multiplication micro-kernel development using AVX2+FMA

* Higher level code that performs L3, L2, and L1 tiling, and uses the 8x8 microkernel for multiplication.

* Testing and verification of the implementation (make sure it is correct)

* Benchmarking against single threaded OpenBLAS.

## Week 2: Tiled matrix multiplication (cont.) (27/03)

- micro-kernel : 3 possibilities, 8x8, 6x16, 4x24 → should be able to use whichever most efficient (?)
- C++ intrinsics
- assembly
- diff execution time for diff sets of parameters (matrix size M k n, f(x1, x2, x3, x4, x5=nb_threads), micro-kernel size etc) → minimize execution time (auto-tuning), find optimal set of parameters for a given matrix size, compare with openBLAS performance for each specific case
- diff algos: greater descent (random starting point, algo tries to find direction to update param, takes step in that direction. If start at bad place → stuck at local minimum : not ideal), stochastic bayesian optimisation (most common, better bc global), ask ai for alternatives, compare
- n executions of kernel for each matrix size, then take median (not average !!) execution time (min and max not useful) → execution script should take care of this. keep deviation small, find optimal value of n : script starts with n = 3 (ex), computes min, max, median and avg, and standard deviation. if standard deviation > 2% n = n+1 (it will stop when standard deviation is small enough)
- make as generic as possible so that it can be used for other kernels (later we will use same tool for other kernels like BLAS1, BLAS2 etc)
- as usual compare w/ openBLAS
- (TASK1 : basic) or multi-threaded parallelism: do task-based (better and easier): compute 1 tile of matrix C (mc x nc) (= 1 openMP task), total = M/nc x N/mc tasks. pb : small matrix  buty many cores → limited parallelism (not enough tasks to occupy all CPU cores) → 2nd version (TASK2 : 3D) : improve → r_task replication : instead of assigning computation of single task, assign r tasks/time. ex : r = 2 → 1st half of tiles multiplied by first task, 2nd half multiplied by 2nd task. task dependencies → partial results → reduction (merge together) → can add more parallelism to kernel ⇒ not limited by task parallelism (= 3D matrix multiplication)
- ⇒ better to generate scripts in python (simpler, more readable)

⇒ understand the code lmao (also how packing & prefetching works, what/how does micro-kernel compute)

Oguz's version: Goals for the week

Make your GEMM implementation compatible with the Goto-BLAS algorithm, which uses L3 cache to hold a tile of B, L2 cache to hold a tile of A, and uses L1 cache as a workspace to fetch smaller tiles of A and B to multiply.

Generate three different micro-kernels (lets say 8x8, 16x6, 4x24), and make the algorithm work with these. Generate assembly versions for each kernel as well. Benchmark the performance in each case.

Add a 2D task-based parallelization; computing each tile (i,j) of C becomes a task. Also add a 3D variant with a constant k where k tasks will cooperate for computing a single tile (i,j) of C (with appropriate dependencies and tile sum/reduction in the end). Compare the multithreaded performance of each case.

Generate generic multivariate performance auto-tuner for your kernels. The script must call the kernel as a function f(x1, x2, x3, ...) where each variable corresponds to a tunable parameter of your kernel (cache tiles sizes or microkernel type for matrix multiplication, loop unrolling factor for example). It should then navigate this parameter search space to find a quasi-optimal parameter set which yields the best performance. Such task is called black-box optimization for which many algorithms exist. The script should be able to use Stochastic Bayesian Optimization andtwo other algorithms of your choice. Compare the performance of all three algorithms (in best performance obtained / number of kernel calls).

## Week 3: Tiled matrix multiplication (cont.) (03/04)

- at each step -> review performance compared to previews steps, instead of just getting a final result at the end (include intermediary results from week 1 and week 2).
- For a fixed matrix size: launch all 3 algos (autotuning: gradient descent, bayesian optimisation and annealing) and get graph: evaluation count on x-axis, and on y-axis kernel performance. see for similar number of evaluations what performance we get. 
- Fix the 'known limitations' from the report in week 2 (no need to focus on Numa - too dirty).
- loop-unrolling for micro-kernel.
- fetch tile, transpose, compute -> try to integrate transposition step into the GEMM itself to see if it improves performance.
- in the load and store part: test non-temporal stores for C and integrate it if it improves performance (writes C matrix directly to memory since we don't need it in the cache anymore). We only need A and B in the cache as we reuse them.

- One of the important optimization techniques is prefetching; as the kernel is performing arithmetic operations on one tile, we can hint the processor to start loading the next tile into the cache. For this, the cache should be able to hold both existing and prefetched tiles simultaneously. Add this to your kernels and test the performance.

- So far, you have been generating and optimizing your kernels interactively with AI. Once you get the optimized implementation, it is preferable to have a complete implementation plan which allows us to generate this kernel from scratch with no a-priori conversation history with the AI. In other words, if you start a new project from scratch and give this implementation plan, AI should be able to generate a code that is as performant as your original implementation (or as close as possible).

- Have the AI agent create agents themself so even this process is automated.


## Week 4: BLAS 1 kernels (only s???? type, excluding strided versions) (10/04)

Generate optimized AVX2 kernels for: sscal, scopy, sswap, saxpy, sdot, snrm2, sasum, isamax, srot.

Add task-based parallelism as before.

Aim to have a good implementation plan for one of the kernels, which should give a good baseline for other kernels since they all have similarities.

You can assume incX and incY are both 1 for simplicity.

Make sure the signature of all your function exactly follows Fortran BLAS signature (sscal(...)) as you will later put all these into a library that will replace OpenBLAS in benchmark tests.

Create a tester that automatically generates many test cases for each kernel (including edge/patological cases) and tests them thoroughly, by comparing against OpenBLAS.

FOR RESULTS: always compare with the best results on each size (ex: what percentage of best results did we reach?). Always keep OPENBLAS comparison in results table. Have intermediate results FOR EVERY SINGLE LITTLE TWEAK.

FOR THE IMPLEMENTATION PLAN: Have an AI agent generate a new agent and loop over the plan to keep improving on its results, setting a performance-goal to reach (or max number of iterations). the master AI agent compares results of the new agent with our results and decide to continue looping or stop. Then generate a report.

## Week 5: BLAS 2 kernels (only s???? type, excluding strided versions) (17/04)

For the TASK-BASED option: use a reduction tree instead to make the reduction part much more efficient bc rn we have barriers that slow everything down. Tasks might be able to beat omp_for loop with more threads (bc of granularity of tasks). try with r=2 and r=4. Also why do we use fma in the reduction step? Investigate -> there might be a more efficient way.

Also, looks like our "3D-parallel" isn't really 3D -> make it ACTUALLY 3D.

In BLAS1: if incX or incY is not 1 (for the functions where that's relevant), we can use a more efficient operation. (check if we're already doing that, if not, implement it).

Have test data in slides as well: what cases we created (sizes, strides, alpha, data types, edge cases etc). Don't just test random stuff, test hot spots => why do we test what we test?

NEW IMPLEMENTATIONS:
- Generate optimized AVX2 kernels for: sgemv, ssymv, strmv, strsv, sger, ssyr, ssyr2

- Add task-based parallelism as before.

- You can assume incX and incY are both 1 for simplicity. If you desire you can also cover the cases with incX or incY >1.

- Make sure the signature of all your function exactly follows Fortran BLAS signature (sgemv_(...)) and not CBLAS signature (cblas_sgemv(...)) as you will later put all these into a library that will replace OpenBLAS in benchmark tests.

- start testing also on 16 threads (to see more scalability)

## Week 6: Other BLAS 3 kernels (trsm, syrk, trmm) (tentative) (24/04)

If you are motivated, you can also optionally try to generate optimized kernels for strsm, ssyrk, and/or strmm.

## Week 7: TER Presentations (date to be announced later)


# Evaluation

Your course grade will be calculated as follows:

1. Project implementation
* You should send your completed lab assignment completed to my university mail (oguz.kaya[at]universite-paris-saclay.fr) with subject line "M1QDCSTERHPC PROJECT Firstname1 LASTNAME1 Firstname2 LASTNAME2" (e.g., M1QDCSTERHPC PROJECT Jean-François DURAND Julie DUFONT).
* In your e-mail, please attach all your project content in a single .zip file (and make sure it does not exceed 20MB).
* If you have a partner/binome for the lab assignment, please send me **one e-mail per group**.
* You should send your work no later than **08/05 Thursday 23:59:59**

2. Project report
* You will write a 20-30 page report on your implementation. There is no fixed format, but make sure to use 11pt policy, single column page format, margins no less than 2cms.
* The report should ideally contain following points:
* A short introduction outlining the project goals and background (no more than 3 pages).
* For each BLAS kernel (or group of kernels), I would like to see the approaches you tried (and succeeded/failed) using AI-driven development. What worked, what did not work, how much improvement did you get applying each optimization, when did AI failed so you had to intervene, when it succeded, etc.
* Experimental section that compares your code's performance to OpenBLAS library
* You should send your completed lab assignment completed to my university mail (oguz.kaya[at]universite-paris-saclay.fr) with subject line "M1QDCSTERHPC REPORT Firstname1 LASTNAME1 Firstname2 LASTNAME2" (e.g., M1QDCSTERHPC REPORT Jean-François DURAND Julie DUFONT).
* You should send your work no later than **08/05 Thursday 23:59:59**

3. Project presentation
* Project presentations will take place together with other individual TER projects, sometime in May. You will receive an e-mail from the TER coordinator (Pablo Arnault) regarding this.
* You will have 15 minutes to present + 5-10 minutes for questions.
* In the presentation, you will not have time to present all your results. You should rather focus on your experience with AI-driven HPC development, what worked, what did not work, and lessons learned, together with some major results (of GEMM kernel for example).

# Tips and advices

* Since all of you will be sharing a single machine, it is important to make a fair use of the CPU resources. In particular, try to optimize the single threaded version of the code first. Once it is efficient, you can run the multi-core benchmarks. For multi-core benchmarks, limit the thread count to 8 for most runs. Once you have a production-ready code, you can benchmark up to 16 threads.
* When using Antigravity, try to use the Gemini Flash model as much as possible; it is remarkably good and you have much higher credit limit. You can save Gemini Pro credits for more critical optimization/organization/code generation tasks. Your credit limit resets about every 4 hours.

# References

* https://github.com/OpenMathLib/OpenBLAS

* https://antigravity.google
