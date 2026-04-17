# Introduction to High-Performance Matrix Multiplication

Welcome! If you're new to the world of High-Performance Computing (HPC) or just wondering why so much effort is put into multiplying grids of numbers, this guide is for you.

We will explain what we are doing, why the obvious way is too slow, and the tricks we use to make it lightning-fast.

---

## 1. What are we doing? (The "SGEMM" Problem)

**SGEMM** stands for **S**ingle-precision **G**eneral **M**atrix **M**ultiplication.

- **Matrix:** Just a grid of numbers. An image is a matrix of pixels; a 3D model is a matrix of coordinates. Neural networks are essentially giant webs of matrices.
- **Single-precision:** This just means we are using standard floating-point numbers (numbers with decimals, like `3.14159`) that take up 32 bits of computer memory.
- **The specific math:** We are calculating `C = A * B`. 
  To get a single number in the output grid `C`, we take a row from grid `A` and a column from grid `B`, multiply their matching numbers together, and add them all up.

### Why does it take so long?
If you have a 1000x1000 matrix `A` and a 1000x1000 matrix `B`, the output `C` is also 1000x1000. That's 1 million numbers to calculate.
To calculate *just one* of those numbers, you have to do 1000 multiplications and 1000 additions. 
So, for the whole grid, your computer has to do **2 billion computations**. 

---

## 2. The Computer's Brain: Why Memory Speed is the Real Enemy

You might think: *"Computers are fast! 2 billion calculations is nothing."* 
And you'd be right! The CPU (the brain of the computer) can do billions of calculations per second. 

**The real problem isn't doing the math; it's getting the numbers from memory to the CPU fast enough.**

Think of a chef in a kitchen:
1. **Registers (The Chef's Hands):** Instantaneous. The chef is holding the ingredients and chopping them right now. 
2. **L1 Cache (The Cutting Board):** Very fast, but tiny. Holds what the chef needs for the next few seconds.
3. **L2 Cache (The Countertop):** Fast, but a bit further away. Holds more ingredients.
4. **L3 Cache (The Fridge):** Slower. Holds enough for the whole recipe.
5. **RAM / Main Memory (The Grocery Store):** Huge, but incredibly slow. If the chef has to drive to the store for every single onion, cooking will take forever.

**The "Naive" Approach:**
If you write a simple 3-line loop in programming to multiply matrices, the computer ends up driving to the grocery store (RAM) for almost every single number. The CPU spends 99% of its time waiting for numbers to arrive, doing literally nothing. This is called a **Cache Miss**.

---

## 3. The Solution: Tiling and Packing

To fix the grocery store problem, we use a strategy often called the **BLIS framework**.

### Trick #1: Cache Blocking (Tiling)
Instead of trying to multiply the whole massive grid at once, we chop matrices `A` and `B` into smaller sub-grids, or "tiles" (often labeled `MC`, `KC`, and `NC`).
We carefully choose the sizes of these tiles so that:
- One tile exactly fills up the Fridge (L3 Cache).
- A smaller sub-tile exactly fills up the Countertop (L2 Cache).
- An even smaller piece goes to the Cutting Board (L1 Cache).

Because we keep the data in the "kitchen" until we are completely done with it, we rarely have to visit the grocery store (RAM). 

### Trick #2: Packing
When you read a book, you read letters from left to right. Memory works the same way; it is much faster to read numbers that are next to each other.
But in matrix math, we often have to read "down a column," which means skipping around in memory. 
To fix this, we **pack** the data: we literally copy the sub-grids into a new, temporary area of memory where the numbers are rearranged sequentially. Now the CPU can read them linearly at top speed.

---

## 4. The Micro-Kernel and AVX (SIMD)

At the very bottom of our whole system, working entirely in the "Chef's Hands" (Registers), is the **Micro-Kernel**. This is a tiny, highly-optimized piece of code that does the actual math on a small chunk of numbers (like a 6x16 grid).

### SIMD (Single Instruction, Multiple Data)
Normally, a CPU does one math problem at a time: `2 * 3 = 6`. 
Modern CPUs have wide lanes called **AVX2**. This allows the CPU to load 8 numbers in a row, load another 8 numbers, and multiply all 8 pairs together simultaneously in a single clock cycle! 
It's the difference between carrying groceries in one hand vs. using a massive cart.

In our project, we have a micro-kernel named `6x16`. It takes 6 rows from A, 16 columns from B, and calculates 96 results at once using these wide AVX lanes effortlessly.

### Trick #3: Latency Hiding (The "Multiple Bowls" Strategy)
Even with AVX, a single calculation takes a few clock cycles to complete. If the Chef waits for the first bowl of soup to heat up before doing anything else, they are wasting time. 
Instead, we use **multiple independent accumulators**. The Chef effectively works on 4 or 12 "bowls" at once. While Bowl #1 is being processed by the hardware, the CPU is already starting the work for Bowl #2. This keeps the CPU's execution units busy every single nanosecond.

---

## 5. The Macro-Kernel: Preparing the Kitchen

While the Micro-Kernel does the chopping, the **Macro-Kernel** is the sous-chef who manages the countertop (L2/L3 Caches).

### Software Prefetching (Looking Ahead)
The CPU is so fast that it often outruns its own cache. To prevent this, we use **Software Prefetching**. 
While the Chef is working on the current set of ingredients, they are already reaching out to the Fridge (L3) to pull the *next* set of ingredients onto the Countertop (L2). By the time the current job is done, the next ingredients are already sitting right there, cold and ready.

### Non-Temporal (NT) Stores (Bypassing the Crowded Counter)
When we finally finish a massive calculation, we need to write the result back to the Grocery Store (RAM). 
Normally, the CPU tries to put everything on the Countertop (L2 Cache) first "just in case" you need it again. But matrix results are often so large they would shove all your useful ingredients off the counter!
We use **Non-Temporal Stores** to tell the CPU: *"Don't put this in the cache; I won't need it again soon. Send it straight to the Grocery Store."* This keeps our fast caches clean and ready for the next set of ingredients.

---

## 6. Working Together: Multi-threading

What if we have multiple chefs (CPU Cores)? We need to split the work so they don't step on each other's toes.

### TASK 1 (2D Parallelism)
Imagine drawing a grid over the final output matrix `C`. We simply tell Core #1 to compute the top-left square, Core #2 to compute the top-right, and so on. This scales brilliantly because the cores don't need to talk to each other.

### TASK 2 (3D Parallelism with K-Replication)
Sometimes, if the matrix is very small but you have a massive CPU (like 16 or 32 cores), there simply aren't enough "squares" in the output matrix `C` to give everyone a job.
Instead of leaving chefs idle, we use **3D Parallelism**. 
- We slice the matrices in the "Z" direction (the K dimension—the inner chunks we are multiplying). 
- Chef #1 computes the first half of a tile's math, and Chef #2 computes the second half.
- They both output "partial" answers.
- At the very end, we use a lightning-fast vector addition step (a **Reduction**) to squish their partial answers together into the final number.

### Adaptive Tiling (Load Balancing)
Even the best chefs work at different speeds if the recipe is complex. If we just give everyone one giant pile of work, some might finish early and sit idle. 
Our system uses **Adaptive MC scaling**. If the matrix is small or we have many cores, we automatically shrink the job sizes ("tiles") so there are more jobs to go around. This ensures that every Chef is always busy until the very last second.

### Loops vs. Tasks (The Factory vs. The Team)
There are two ways to tell the Chefs what to do:
1. **Loop-based Parallelism (`omp for`)**: Like a rigid factory assembly line. It is incredibly fast for standard, regular work. 
2. **Task-based Parallelism (`omp task`)**: Like a flexible emergency response team. Every job is a "ticket" that any available Chef can pick up. 
While Tasks are more flexible, they have more "paperwork" (overhead). We benchmarked both and found that for predictable grids like matrix math, the **Loop-based factory** is usually the winner!

---

## 7. BLAS 1: The Basic Building Blocks

While SGEMM is the "heavy lifter" for big grids, every math library needs basic tools for smaller jobs—like scaling a single row or adding two vectors. These are called **BLAS Level 1** operations.

- **The Knives and Spoons:** Operations like `sscal` (multiplying a vector by a constant) or `saxpy` (adding a multiple of one vector to another).
- **AVX Optimized:** Just like our big matrix kernel, these tiny tools use AVX2 "bulk carrier" instructions.
- **Latency Hiding:** Even for a simple addition, we use multiple independent accumulators so the CPU can calculate different parts of the vector simultaneously using both "hands."

---

## 8. Auto-Tuning: Finding the Perfect Setup

Different computers have different sized "fridges" and "countertops" (Caches). A tile size of `120x512` might run brilliantly on an Intel CPU, but terribly on an AMD CPU.

Instead of guessing, we wrote **Auto-tuning algorithms** (using logic inspired by AI like Gradient Descent or Bayesian Optimization). 
Our Python scripts act like a director: they ask the C code to try dozens of different tile sizes (`MC`, `KC`, `NC`), measure exactly how many milliseconds each takes, and automatically narrow down on the absolute "sweet spot" for whatever machine it's currently running on.

---

## 9. Compatibility (Speaking International Languages)

Scientific software is often written in many languages, like **Fortran** or **Python**. To make our library useful to everyone, we implemented the **Fortran ABI**. This is like a set of international translation rules that allow old, powerful scientific programs to talk to our modern, lightning-fast C code without any confusion.

---

### Summary
By managing memory carefully (**Tiling and Packing**), doing math in bulk (**AVX2**), hiding hardware latency (**Independent Accumulators**), effectively using all CPU cores (**Adaptive Multi-threading**), and automatically adapting to the hardware (**Auto-tuning**), this project takes operations that would normally take a computer several seconds and completes them in a fraction of a millisecond. 

Whether it's the massive heavy-lifting of SGEMM or the swift utilities of BLAS 1, this library is built to squeeze every drop of performance out of modern silicon.
