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

---

## 5. Working Together: Multi-threading

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

---

## 6. Auto-Tuning: Finding the Perfect Setup

Different computers have different sized "fridges" and "countertops" (Caches). A tile size of `120x512` might run brilliantly on an Intel CPU, but terribly on an AMD CPU.

Instead of guessing, we wrote **Auto-tuning algorithms** (using logic inspired by AI like Gradient Descent or Bayesian Optimization). 
Our Python scripts act like a director: they ask the C code to try dozens of different tile sizes (`MC`, `KC`, `NC`), measure exactly how many milliseconds each takes, and automatically narrow down on the absolute "sweet spot" for whatever machine it's currently running on.

---

### Summary
By managing memory carefully (Tiling and Packing), doing math in bulk (AVX2), effectively using all CPU cores (Multi-threading), and adapting to the hardware (Auto-tuning), this project takes an operation that would normally take a computer several seconds and completes it in a fraction of a millisecond, outperforming commercial libraries like OpenBLAS!
