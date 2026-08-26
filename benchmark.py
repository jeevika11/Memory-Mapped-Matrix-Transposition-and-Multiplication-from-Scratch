import os
import time
import numpy as np

from matrix_engine import (
    transpose_memmap,
    block_matrix_multiply
)


SHAPE = (2000, 2000)
BLOCK_SIZE = 256

INPUT = "benchmark_A.dat"
TRANSPOSE = "benchmark_AT.dat"
RESULT = "benchmark_result.dat"


print("=" * 60)
print("TASK 4 BLOCK ENGINE BENCHMARK")
print("=" * 60)

print(f"Shape      : {SHAPE}")
print(f"Block size : {BLOCK_SIZE}")


# ------------------------------------------------
# Transpose
# ------------------------------------------------

print("\nBLOCK TRANSPOSE")

start = time.perf_counter()

transpose_memmap(
    INPUT,
    TRANSPOSE,
    SHAPE,
    block_size=BLOCK_SIZE
)

transpose_time = time.perf_counter() - start

print(
    f"Transpose time: "
    f"{transpose_time:.4f} seconds"
)


# ------------------------------------------------
# Multiplication
# ------------------------------------------------

print("\nBLOCK MULTIPLICATION")

start = time.perf_counter()

block_matrix_multiply(
    INPUT,
    TRANSPOSE,
    RESULT,
    SHAPE,
    SHAPE,
    block_size=BLOCK_SIZE
)

multiplication_time = (
    time.perf_counter() - start
)

print(
    f"Multiplication time: "
    f"{multiplication_time:.4f} seconds"
)


# ------------------------------------------------
# Verify
# ------------------------------------------------

A = np.memmap(
    INPUT,
    dtype=np.float64,
    mode="r",
    shape=SHAPE
)

AT = np.memmap(
    TRANSPOSE,
    dtype=np.float64,
    mode="r",
    shape=SHAPE
)

C = np.memmap(
    RESULT,
    dtype=np.float64,
    mode="r",
    shape=SHAPE
)

expected = (
    np.asarray(A[:100])
    @
    np.asarray(AT[:, :100])
)

actual = np.asarray(
    C[:100, :100]
)

difference = np.max(
    np.abs(expected - actual)
)

print("\nVerification")
print(
    f"Maximum difference: "
    f"{difference:.10f}"
)

if difference < 1e-8:
    print("Verification: PASS")
else:
    print("Verification: FAIL")


del A
del AT
del C

print("\n" + "=" * 60)
print("BENCHMARK COMPLETED")
print("=" * 60)