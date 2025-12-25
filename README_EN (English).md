# PyWave: Linear Complexity Neuro-Framework

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey)

**PyWave** is a research-grade C++/Python library that replaces the quadratic $O(N^2)$ complexity of standard Matrix Multiplication (MatMul) with linear $O(N)$ Phase Interference dynamics.

Designed for ultra-long context processing where Transformers fail.

## 🚀 Key Features
- **Phase Interference Core**: Replaces Self-Attention with physical wave dynamics.
- **True Linear Scaling**: 18x faster than PyTorch on 8k context (CPU-only).
- **Dynamic ADS**: Adaptive Decay System for biological-like plasticity.
- **No GPU Required**: Highly optimized AVX-512/AMX kernels.

## 📊 Benchmarks (The Quadratic Wall)

Comparison against PyTorch (MKL Backend) on synthetic tasks.
*Hardware: [Укажите ваш CPU, например: Intel Core i9-13900K / AMD Ryzen 9 7950X]*

| Dimension (N) | PyTorch (ms) | PyWave (ms) | Speedup | Logic |
|:--------------|:-------------|:------------|:--------|:------|
| 128           | 0.53         | 0.24        | **2.19x**| Overhead Win |
| 512           | 0.40         | 0.61        | 0.65x   | Optimization Gap |
| 1024          | 0.91         | 1.16        | 0.78x   | Transition Zone |
| 2048          | 5.53         | 2.10        | **2.63x**| **Break Point** |
| 4096          | 24.54        | 3.79        | **6.48x**| Linear Scaling |
| 8192          | 118.61       | 6.55        | **18.10x**| **Quadratic Wall** |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- C++ Compiler (g++ / clang / MSVC) with OpenMP support.

### 1. Clone
```bash
git clone [https://github.com/corpuscul-wave/PyWave.git](https://github.com/corpuscul-wave/PyWave.git)
cd PyWave
