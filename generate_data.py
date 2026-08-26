"""
generate_data.py
----------------------------------------------------
Task 4: Disk-backed Matrix Generation

Generates an ultra-large random matrix directly
to disk using NumPy memmap.

The complete matrix is NEVER loaded into RAM.

Required technology:
    - NumPy
    - NumPy memmap
    - Python file I/O
"""

import os
import time
import shutil
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DTYPE = np.float64

# Generate only 256 rows at a time.
GENERATION_BLOCK_SIZE = 256

# Task 4 requires approximately 15 GB of raw data.
FINAL_SHAPE = (44869, 44869)

OUTPUT_FILE = "matrix_A_15GB.dat"

# Three disk-backed files will eventually exist:
# 1. Input
# 2. Transpose
# 3. Result
FILES_REQUIRED = 3


# ============================================================
# DISK SPACE
# ============================================================

def get_disk_space(path="."):
    """
    Return disk-space information in GB.
    """

    total, used, free = shutil.disk_usage(path)

    return {
        "total": total / (1024 ** 3),
        "used": used / (1024 ** 3),
        "free": free / (1024 ** 3)
    }


# ============================================================
# SIZE CALCULATION
# ============================================================

def calculate_size(shape, dtype=np.float64):
    """
    Calculate the exact size of a matrix.
    """

    rows, cols = shape

    total_bytes = (
        rows
        * cols
        * np.dtype(dtype).itemsize
    )

    size_mb = total_bytes / (1024 ** 2)
    size_gb = total_bytes / (1024 ** 3)

    return total_bytes, size_mb, size_gb


# ============================================================
# MEMORY CHECK
# ============================================================

def check_generation_memory(
    cols,
    block_size,
    dtype=np.float64
):
    """
    Check the temporary block size.

    Only one generated block is explicitly created
    as a normal NumPy array.
    """

    block_bytes = (
        block_size
        * cols
        * np.dtype(dtype).itemsize
    )

    block_mb = block_bytes / (1024 ** 2)

    print("\n" + "-" * 70)
    print("GENERATION MEMORY CHECK")
    print("-" * 70)

    print(
        f"Generation block : "
        f"{block_size} x {cols}"
    )

    print(
        f"Temporary block   : "
        f"{block_mb:.2f} MB"
    )

    print(
        "512 MB RAM budget : "
        f"{512:.2f} MB"
    )

    if block_mb >= 512:

        print(
            "\nERROR: Generation block is too large."
        )

        return False

    print(
        "\nGeneration memory check : PASS"
    )

    return True


# ============================================================
# MATRIX GENERATION
# ============================================================

def generate_matrix(
    filename,
    shape,
    dtype=np.float64,
    seed=42,
    block_size=256
):
    """
    Generate a random matrix directly to disk.

    Data is generated row-block by row-block.
    """

    rows, cols = shape

    # --------------------------------------------------------
    # Calculate matrix size
    # --------------------------------------------------------

    total_bytes, size_mb, size_gb = calculate_size(
        shape,
        dtype
    )

    print("=" * 70)
    print("MEMORY-MAPPED DATA GENERATION")
    print("=" * 70)

    print(
        f"Output file      : {filename}"
    )

    print(
        f"Matrix shape     : {shape}"
    )

    print(
        f"Data type        : {dtype}"
    )

    print(
        f"Block size       : {block_size} rows"
    )

    print(
        f"Matrix size      : "
        f"{size_mb:.2f} MB "
        f"({size_gb:.4f} GiB)"
    )

    # --------------------------------------------------------
    # Check generation memory
    # --------------------------------------------------------

    if not check_generation_memory(
        cols,
        block_size,
        dtype
    ):
        return False

    # --------------------------------------------------------
    # Check disk space
    # --------------------------------------------------------

    disk = get_disk_space()

    # We need space for input + transpose + result.
    required_disk_gb = (
        size_gb
        * FILES_REQUIRED
    )

    print("\n" + "-" * 70)
    print("DISK SPACE CHECK")
    print("-" * 70)

    print(
        f"One matrix        : "
        f"{size_gb:.4f} GiB"
    )

    print(
        f"Files required    : "
        f"{FILES_REQUIRED}"
    )

    print(
        f"Estimated total   : "
        f"{required_disk_gb:.4f} GiB"
    )

    print(
        f"Available disk    : "
        f"{disk['free']:.4f} GiB"
    )

    if disk["free"] < required_disk_gb:

        print(
            "\nERROR: Not enough disk space."
        )

        print(
            f"Required : "
            f"{required_disk_gb:.2f} GiB"
        )

        print(
            f"Available: "
            f"{disk['free']:.2f} GiB"
        )

        print(
            "\nFree disk space before "
            "starting the 15 GB task."
        )

        return False

    print(
        "\nDisk space check : PASS"
    )

    # --------------------------------------------------------
    # Remove existing file
    # --------------------------------------------------------

    if os.path.exists(filename):

        existing_size = (
            os.path.getsize(filename)
            / (1024 ** 3)
        )

        print()
        print(
            f"Existing file found:"
            f" {existing_size:.4f} GiB"
        )

        answer = input(
            "Replace existing file? [y/N]: "
        ).strip().lower()

        if answer != "y":

            print(
                "Generation cancelled."
            )

            return False

        os.remove(filename)

    # --------------------------------------------------------
    # Create memmap
    # --------------------------------------------------------

    print()
    print(
        "Creating memory-mapped file..."
    )

    matrix = np.memmap(
        filename,
        dtype=dtype,
        mode="w+",
        shape=shape
    )

    # --------------------------------------------------------
    # Random generator
    # --------------------------------------------------------

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Block generation
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("GENERATING MATRIX")
    print("-" * 70)

    start_time = time.perf_counter()

    total_blocks = (
        rows
        + block_size
        - 1
    ) // block_size

    block_number = 0

    try:

        for row_start in range(
            0,
            rows,
            block_size
        ):

            row_end = min(
                row_start + block_size,
                rows
            )

            rows_in_block = (
                row_end
                - row_start
            )

            # ------------------------------------------------
            # Only this block is a normal NumPy array.
            # ------------------------------------------------

            block = rng.random(
                (
                    rows_in_block,
                    cols
                ),
                dtype=dtype
            )

            # Write directly into memmap.
            matrix[
                row_start:row_end
            ] = block

            block_number += 1

            # Release temporary RAM block.
            del block

            # Flush periodically.
            if (
                block_number % 10 == 0
                or row_end == rows
            ):

                matrix.flush()

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                percentage = (
                    row_end
                    / rows
                ) * 100

                print(
                    f"Progress: "
                    f"{percentage:6.2f}% | "
                    f"Block: "
                    f"{block_number:,}/"
                    f"{total_blocks:,} | "
                    f"Rows: "
                    f"{row_end:,}/"
                    f"{rows:,} | "
                    f"Elapsed: "
                    f"{elapsed:.1f}s"
                )

    except Exception as error:

        print()
        print(
            "ERROR during data generation:"
        )

        print(error)

        # Flush what has already been written.
        matrix.flush()

        del matrix

        print(
            "\nPartial file has been left on disk."
        )

        print(
            "Delete it before retrying."
        )

        return False

    # --------------------------------------------------------
    # Final flush
    # --------------------------------------------------------

    matrix.flush()

    del matrix

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Verify generated file
    # --------------------------------------------------------

    actual_bytes = os.path.getsize(
        filename
    )

    actual_gib = (
        actual_bytes
        / (1024 ** 3)
    )

    print()
    print("=" * 70)
    print("DATA GENERATION COMPLETED")
    print("=" * 70)

    print(
        f"Output file     : "
        f"{filename}"
    )

    print(
        f"Expected size   : "
        f"{size_gb:.4f} GiB"
    )

    print(
        f"Actual size     : "
        f"{actual_gib:.4f} GiB"
    )

    print(
        f"Generation time : "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"Blocks written  : "
        f"{block_number:,}"
    )

    if actual_bytes == total_bytes:

        print(
            "\nFile size verification : PASS"
        )

        return True

    print(
        "\nFile size verification : FAIL"
    )

    return False


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    generate_matrix(
        filename=OUTPUT_FILE,
        shape=FINAL_SHAPE,
        dtype=DTYPE,
        seed=42,
        block_size=GENERATION_BLOCK_SIZE
    )