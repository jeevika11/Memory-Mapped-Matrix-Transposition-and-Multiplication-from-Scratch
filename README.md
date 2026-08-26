Memory-Mapped-Matrix-Transposition-and-Multiplication-from-Scratch
# Memory-Mapped Matrix Transposition and Multiplication from Scratch

## Task 4 – Out-of-Core Matrix Processing Using NumPy Memmap

This project implements large-scale matrix processing using **NumPy memory mapping (`memmap`)** and **block-wise computation**.

The main objective is to process matrices that can exceed the available physical RAM by keeping the complete dataset on disk and loading only small blocks into memory when required.

---

## 🎯 Objective

The objective of Task 4 is to implement:

- Disk-backed matrix generation
- Memory-mapped matrix storage
- Block-wise matrix transposition
- Block-wise matrix multiplication
- Disk-backed result storage
- Numerical verification
- Memory-conscious processing

The implementation is designed for an **out-of-core processing model** under a strict **512 MB RAM budget**.

---

## 🛠️ Technology Stack

- Python
- NumPy
- NumPy `memmap`
- Python File I/O

No external numerical computing libraries are required.

---

## 📂 Project Structure

```text
Memory-Mapped-Matrix-Transposition-and-Multiplication-from-Scratch/
│
├── README.md
├── generate_data.py
├── matrix_engine.py
├── task4.py
├── benchmark.py
└── .gitignore
