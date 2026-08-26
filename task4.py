"""
task4.py
----------------------------------------------------
Task 4: Memory-Mapped Matrix Transposition and
Block Matrix Multiplication

Objective:
    Process ultra-large disk-backed matrices using
    NumPy memmap and block-wise operations while
    keeping the working memory below a 512 MB budget.
"""

import os
import time
import shutil
import numpy as np

from matrix_engine import (
    transpose_memmap,
    block_matrix_multiply
)


# ====================================================
# CONFIGURATION
# ====================================================

RAM_LIMIT_MB = 512

# 15 GB-class matrix
#
# 44869 x 44869 x 8 bytes
# ≈ 15.00 GiB
#
# This is approximately 15 GB of raw float64 data.

SHAPE = (44869, 44869)

INPUT_FILE = "matrix_A_15GB.dat"
TRANSPOSE_FILE = "matrix_A_transposed_15GB.dat"
RESULT_FILE = "matrix_result_15GB.dat"

DTYPE = np.float64

# 256 x 256 blocks
#
# Three blocks:
# A block
# B block
# Result block
#
# 3 * 256 * 256 * 8 bytes
# ≈ 1.5 MB
#
# Well below the 512 MB limit.

BLOCK_SIZE = 256

SAMPLE_SIZE = 100


# ====================================================
# SIZE CALCULATION
# ====================================================

def calculate_matrix_size(shape, dtype=np.float64):

    rows, cols = shape

    total_bytes = (
        rows
        * cols
        * np.dtype(dtype).itemsize
    )

    size_mb = total_bytes / (1024 ** 2)
    size_gb = total_bytes / (1024 ** 3)

    return size_mb, size_gb


# ====================================================
# DISK SPACE
# ====================================================

def get_available_disk_space(path="."):

    _, _, free = shutil.disk_usage(path)

    return free / (1024 ** 3)


def check_disk_space(
    shape,
    dtype=np.float64,
    files_required=3
):
    """
    Check disk space required for:

        1. Input matrix
        2. Transposed matrix
        3. Result matrix
    """

    matrix_mb, matrix_gb = calculate_matrix_size(
        shape,
        dtype
    )

    required_gb = matrix_gb * files_required
    available_gb = get_available_disk_space()

    print("\n" + "-" * 60)
    print("DISK SPACE CHECK")
    print("-" * 60)

    print(
        f"Single matrix size : "
        f"{matrix_mb:.2f} MB "
        f"({matrix_gb:.2f} GB)"
    )

    print(
        f"Required space     : "
        f"{required_gb:.2f} GB"
    )

    print(
        f"Available space    : "
        f"{available_gb:.2f} GB"
    )

    if available_gb < required_gb:

        print("\nERROR: Not enough disk space.")

        print(
            f"Required : {required_gb:.2f} GB"
        )

        print(
            f"Available: {available_gb:.2f} GB"
        )

        return False

    print("\nDisk space check : PASS")

    return True


# ====================================================
# RAM BUDGET
# ====================================================

def check_block_memory(
    block_size,
    dtype=np.float64
):
    """
    Estimate memory used by the temporary blocks.

    Three blocks are considered:

        A_block
        B_block
        C_block
    """

    itemsize = np.dtype(dtype).itemsize

    block_bytes = (
        block_size
        * block_size
        * itemsize
    )

    estimated_bytes = block_bytes * 3

    estimated_mb = (
        estimated_bytes
        / (1024 ** 2)
    )

    print("\n" + "-" * 60)
    print("512 MB RAM BUDGET CHECK")
    print("-" * 60)

    print(
        f"Block size         : "
        f"{block_size} x {block_size}"
    )

    print(
        f"Data type          : "
        f"{dtype}"
    )

    print(
        f"One block size     : "
        f"{block_bytes / (1024 ** 2):.2f} MB"
    )

    print(
        f"Estimated working  : "
        f"{estimated_mb:.2f} MB"
    )

    print(
        f"RAM budget         : "
        f"{RAM_LIMIT_MB} MB"
    )

    if estimated_mb > RAM_LIMIT_MB:

        print(
            "\nERROR: Block size exceeds "
            "the 512 MB RAM budget."
        )

        return False

    print("\nRAM budget check   : PASS")

    return True


# ====================================================
# INPUT FILE VALIDATION
# ====================================================

def check_input_file(
    filename,
    shape,
    dtype=np.float64
):

    if not os.path.exists(filename):

        print(
            f"\nERROR: {filename} not found."
        )

        print(
            "\nRun generate_data.py first "
            "to create the 15 GB dataset."
        )

        return False

    expected_bytes = (
        shape[0]
        * shape[1]
        * np.dtype(dtype).itemsize
    )

    actual_bytes = os.path.getsize(filename)

    print("\n" + "-" * 60)
    print("INPUT FILE CHECK")
    print("-" * 60)

    print(
        f"Expected size : "
        f"{expected_bytes / (1024 ** 3):.4f} GB"
    )

    print(
        f"Actual size   : "
        f"{actual_bytes / (1024 ** 3):.4f} GB"
    )

    if actual_bytes != expected_bytes:

        print(
            "\nERROR: Input file size does not "
            "match the configured matrix shape."
        )

        print(
            "\nExpected bytes : "
            f"{expected_bytes}"
        )

        print(
            "Actual bytes   : "
            f"{actual_bytes}"
        )

        print(
            "\nMake sure generate_data.py and "
            "task4.py use the same shape."
        )

        return False

    print("\nInput file check : PASS")

    return True


# ====================================================
# TRANSPOSE VERIFICATION
# ====================================================

def verify_transpose(
    input_file,
    transpose_file,
    shape,
    dtype=np.float64,
    sample_size=100
):

    print("\n" + "-" * 60)
    print("TRANSPOSE VERIFICATION")
    print("-" * 60)

    original = np.memmap(
        input_file,
        dtype=dtype,
        mode="r",
        shape=shape
    )

    transposed = np.memmap(
        transpose_file,
        dtype=dtype,
        mode="r",
        shape=(shape[1], shape[0])
    )

    sample_size = min(
        sample_size,
        shape[0],
        shape[1]
    )

    # Only load a tiny 100 x 100 sample.
    transpose_expected = np.asarray(
        original[
            :sample_size,
            :sample_size
        ]
    ).T

    transpose_actual = np.asarray(
        transposed[
            :sample_size,
            :sample_size
        ]
    )

    difference = np.max(
        np.abs(
            transpose_actual
            - transpose_expected
        )
    )

    print(
        f"Sample size       : "
        f"{sample_size} x {sample_size}"
    )

    print(
        f"Maximum Difference : "
        f"{difference:.10f}"
    )

    if difference < 1e-10:

        print(
            "Transpose Check : PASS"
        )

    else:

        print(
            "Transpose Check : FAIL"
        )

    del original
    del transposed

    return difference < 1e-10


# ====================================================
# MULTIPLICATION VERIFICATION
# ====================================================

def verify_multiplication(
    input_file,
    transpose_file,
    result_file,
    shape,
    dtype=np.float64,
    sample_size=100
):

    print("\n" + "-" * 60)
    print("MATRIX MULTIPLICATION VERIFICATION")
    print("-" * 60)

    original = np.memmap(
        input_file,
        dtype=dtype,
        mode="r",
        shape=shape
    )

    transposed = np.memmap(
        transpose_file,
        dtype=dtype,
        mode="r",
        shape=(shape[1], shape[0])
    )

    result = np.memmap(
        result_file,
        dtype=dtype,
        mode="r",
        shape=(shape[0], shape[0])
    )

    sample_size = min(
        sample_size,
        shape[0],
        shape[1]
    )

    # ------------------------------------------------
    # IMPORTANT
    # ------------------------------------------------
    # Only calculate a small 100 x 100 verification
    # result.
    #
    # No complete matrix is loaded into RAM.
    # ------------------------------------------------

    A_sample = np.asarray(
        original[
            :sample_size,
            :
        ]
    )

    B_sample = np.asarray(
        transposed[
            :,
            :sample_size
        ]
    )

    expected = (
        A_sample
        @
        B_sample
    )

    actual = np.asarray(
        result[
            :sample_size,
            :sample_size
        ]
    )

    difference = np.max(
        np.abs(
            actual
            - expected
        )
    )

    print(
        f"Sample size       : "
        f"{sample_size} x {sample_size}"
    )

    print(
        f"Maximum Difference : "
        f"{difference:.10f}"
    )

    if difference < 1e-8:

        print(
            "Multiplication Check : PASS"
        )

    else:

        print(
            "Multiplication Check : FAIL"
        )

    del original
    del transposed
    del result

    del A_sample
    del B_sample
    del expected
    del actual

    return difference < 1e-8


# ====================================================
# FILE INFORMATION
# ====================================================

def print_file_information(
    filenames
):

    print("\n" + "-" * 60)
    print("OUTPUT FILES")
    print("-" * 60)

    for filename in filenames:

        if os.path.exists(filename):

            size_gb = (
                os.path.getsize(filename)
                / (1024 ** 3)
            )

            print(
                f"{filename:<35} "
                f"{size_gb:.4f} GB"
            )


# ====================================================
# MAIN
# ====================================================

def main():

    print("=" * 60)
    print("TASK 4: MEMORY-MAPPED MATRIX ENGINE")
    print("=" * 60)

    # ------------------------------------------------
    # Configuration
    # ------------------------------------------------

    shape = SHAPE

    input_file = INPUT_FILE
    transpose_file = TRANSPOSE_FILE
    result_file = RESULT_FILE

    block_size = BLOCK_SIZE
    dtype = DTYPE

    print("\n" + "-" * 60)
    print("CONFIGURATION")
    print("-" * 60)

    print(
        f"Matrix shape      : {shape}"
    )

    print(
        f"Data type         : {dtype}"
    )

    print(
        f"Block size        : "
        f"{block_size} x {block_size}"
    )

    print(
        f"RAM limit         : "
        f"{RAM_LIMIT_MB} MB"
    )

    matrix_mb, matrix_gb = calculate_matrix_size(
        shape,
        dtype
    )

    print(
        f"Matrix size       : "
        f"{matrix_gb:.4f} GB"
    )

    # ------------------------------------------------
    # Disk Space Check
    # ------------------------------------------------

    if not check_disk_space(
        shape,
        dtype=dtype,
        files_required=3
    ):

        return

    # ------------------------------------------------
    # RAM Budget Check
    # ------------------------------------------------

    if not check_block_memory(
        block_size,
        dtype=dtype
    ):

        return

    # ------------------------------------------------
    # Check Input
    # ------------------------------------------------

    if not check_input_file(
        input_file,
        shape,
        dtype
    ):

        return

    # ------------------------------------------------
    # Memory-Mapped Transposition
    # ------------------------------------------------

    print("\n" + "-" * 60)
    print("BLOCK-WISE TRANSPOSE")
    print("-" * 60)

    start = time.perf_counter()

    transpose_memmap(
        input_file=input_file,
        output_file=transpose_file,
        shape=shape,
        block_size=block_size
    )

    transpose_time = (
        time.perf_counter()
        - start
    )

    print(
        f"Transpose Time : "
        f"{transpose_time:.6f} seconds"
    )

    # ------------------------------------------------
    # Block Matrix Multiplication
    # ------------------------------------------------

    print("\n" + "-" * 60)
    print("BLOCK-WISE MATRIX MULTIPLICATION")
    print("-" * 60)

    start = time.perf_counter()

    block_matrix_multiply(
        a_file=input_file,
        b_file=transpose_file,
        output_file=result_file,
        a_shape=shape,
        b_shape=(shape[1], shape[0]),
        block_size=block_size
    )

    multiplication_time = (
        time.perf_counter()
        - start
    )

    print(
        f"Multiplication Time : "
        f"{multiplication_time:.6f} seconds"
    )

    # ------------------------------------------------
    # Verify Transpose
    # ------------------------------------------------

    transpose_pass = verify_transpose(
        input_file=input_file,
        transpose_file=transpose_file,
        shape=shape,
        dtype=dtype,
        sample_size=SAMPLE_SIZE
    )

    # ------------------------------------------------
    # Verify Matrix Multiplication
    # ------------------------------------------------

    multiplication_pass = verify_multiplication(
        input_file=input_file,
        transpose_file=transpose_file,
        result_file=result_file,
        shape=shape,
        dtype=dtype,
        sample_size=SAMPLE_SIZE
    )

    # ------------------------------------------------
    # File Information
    # ------------------------------------------------

    print_file_information(
        [
            input_file,
            transpose_file,
            result_file
        ]
    )

    # ------------------------------------------------
    # Final Status
    # ------------------------------------------------

    print("\n" + "=" * 60)

    if (
        transpose_pass
        and multiplication_pass
    ):

        print("TASK 4 TEST COMPLETED")
        print("TRANSPOSE CHECK      : PASS")
        print("MULTIPLICATION CHECK : PASS")

    else:

        print("TASK 4 TEST COMPLETED")
        print("CHECKS REQUIRE ATTENTION")

    print("=" * 60)


# ====================================================
# PROGRAM ENTRY
# ====================================================

if __name__ == "__main__":
    main()