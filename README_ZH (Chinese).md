# PyWave v1.0: 线性复杂度计算框架

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey)

**PyWave** 是一个使用 C++ 和 Python 编写的高性能类脑计算库。它旨在克服现代人工智能架构的“平方墙”难题。

## 🚀 核心理念
PyWave 弃用了标准的矩阵乘法 ($O(N^2)$)，转而利用动态环境中的**相位干涉**。这实现了**真正的线性复杂度 $O(N)$**，使得在普通 CPU 上高效处理超长上下文成为可能。

## 📊 基准测试 (仅限 CPU)
与 PyTorch (Intel MKL) 在 8192 维度下的对比：
- **PyTorch**: 118.61 毫秒
- **PyWave**: 7.18 毫秒
- **加速比**: **16.51x** (突破平方复杂度瓶颈)

## 🧠 核心技术
- **相位干涉核心**: 通过波场共振进行计算。
- **动态 ADS**: 遵循“用进废退”原则的自适应记忆衰减系统。
- **随机共振**: 引入朗之万动力学噪声，防止模型陷入死循环。
- **稳态门控**: 防止神经元过热和共振幻觉的自我调节机制。

## 🛠️ 快速开始
