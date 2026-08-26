"""
matrix_engine.py
----------------------------------------------------
Task 4: Memory-Mapped Matrix Operations

Features
--------
- NumPy memmap
- Block-wise matrix transposition
- Block-wise matrix multiplication
- Disk-backed output
- Controlled working memory
- Explicit flushing to disk
"""

import numpy as np


# ============================================================
# BLOCK-WISE TRANSPOSE
# ============================================================

def transpose_memmap(
    input_file,
    output_file,
    shape,
    block_size=256,
    dtype=np.float64
):
    """
    Transpose a large disk-backed matrix block by block.

    The complete matrix is never loaded into RAM.

    Parameters
    ----------
    input_file : str
        Input binary matrix file.

    output_file : str
        Output transposed matrix file.

    shape : tuple
        Original matrix shape (rows, cols).

    block_size : int
        Number of rows/columns processed per block.

    dtype : NumPy dtype
        Matrix data type.
    """

    rows, cols = shape

    # --------------------------------------------------------
    # Open input as memory-mapped array
    # --------------------------------------------------------

    source = np.memmap(
        input_file,
        dtype=dtype,
        mode="r",
        shape=shape
    )

    # --------------------------------------------------------
    # Create output memmap
    # --------------------------------------------------------

    destination = np.memmap(
        output_file,
        dtype=dtype,
        mode="w+",
        shape=(cols, rows)
    )

    print(
        f"Transpose blocks: "
        f"{block_size} x {block_size}"
    )

    # --------------------------------------------------------
    # Process matrix block by block
    # --------------------------------------------------------

    for row_start in range(
        0,
        rows,
        block_size
    ):

        row_end = min(
            row_start + block_size,
            rows
        )

        for col_start in range(
            0,
            cols,
            block_size
        ):

            col_end = min(
                col_start + block_size,
                cols
            )

            # Read only one small block
            block = source[
                row_start:row_end,
                col_start:col_end
            ]

            # Transpose the small block
            destination[
                col_start:col_end,
                row_start:row_end
            ] = block.T

    # --------------------------------------------------------
    # Flush output to disk
    # --------------------------------------------------------

    destination.flush()

    # Release memory mappings
    del source
    del destination


# ============================================================
# BLOCK-WISE MATRIX MULTIPLICATION
# ============================================================

def block_matrix_multiply(
    a_file,
    b_file,
    output_file,
    a_shape,
    b_shape,
    block_size=256,
    dtype=np.float64
):
    """
    Perform out-of-core block matrix multiplication.

    A = (M x K)
    B = (K x N)

    Result:

        C = A @ B

    The complete matrices are never loaded into RAM.
    """

    m, k = a_shape

    k2, n = b_shape

    # --------------------------------------------------------
    # Validate matrix dimensions
    # --------------------------------------------------------

    if k != k2:

        raise ValueError(
            "Matrix dimensions are incompatible: "
            f"A={a_shape}, B={b_shape}"
        )

    # --------------------------------------------------------
    # Open input matrices as memmaps
    # --------------------------------------------------------

    A = np.memmap(
        a_file,
        dtype=dtype,
        mode="r",
        shape=a_shape
    )

    B = np.memmap(
        b_file,
        dtype=dtype,
        mode="r",
        shape=b_shape
    )

    # --------------------------------------------------------
    # Create disk-backed result
    # --------------------------------------------------------

    C = np.memmap(
        output_file,
        dtype=dtype,
        mode="w+",
        shape=(m, n)
    )

    # --------------------------------------------------------
    # Initialize output
    #
    # Only the disk-backed memmap is initialized.
    # No full normal NumPy array is created.
    # --------------------------------------------------------

    C[:] = 0.0

    # --------------------------------------------------------
    # Block matrix multiplication
    # --------------------------------------------------------

    total_row_blocks = (
        m + block_size - 1
    ) // block_size

    completed_row_blocks = 0

    for i in range(
        0,
        m,
        block_size
    ):

        i_end = min(
            i + block_size,
            m
        )

        for j in range(
            0,
            n,
            block_size
        ):

            j_end = min(
                j + block_size,
                n
            )

            # ------------------------------------------------
            # Accumulate C block
            # ------------------------------------------------

            for k_start in range(
                0,
                k,
                block_size
            ):

                k_end = min(
                    k_start + block_size,
                    k
                )

                # --------------------------------------------
                # Read small blocks
                # --------------------------------------------

                A_block = A[
                    i:i_end,
                    k_start:k_end
                ]

                B_block = B[
                    k_start:k_end,
                    j:j_end
                ]

                # --------------------------------------------
                # Small matrix multiplication
                # --------------------------------------------

                C[
                    i:i_end,
                    j:j_end
                ] += A_block @ B_block

            # Flush completed C block to disk
            C.flush()

        completed_row_blocks += 1

        percentage = (
            completed_row_blocks
            / total_row_blocks
        ) * 100

        print(
            f"Multiplication progress: "
            f"{percentage:6.2f}%"
        )

    # --------------------------------------------------------
    # Final flush
    # --------------------------------------------------------

    C.flush()

    # --------------------------------------------------------
    # Release memory mappings
    # --------------------------------------------------------

    del A
    del B
    del C