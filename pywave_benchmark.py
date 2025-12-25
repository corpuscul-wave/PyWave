import ctypes
import numpy as np
import time
import torch
import torch.nn as nn
import os
import sys
import csv
import platform

# === CONFIGURATION ===
LIB_PATH = "libpywave.dll"
BATCH_SIZE = 64
WARMUP_STEPS = 10
TEST_STEPS = 50
DIMENSIONS = [128, 512, 1024, 2048, 4096, 8192] # The "Quadratic Wall" test
RESULTS_FILE = "benchmark_results.csv"

# === 1. PYWAVE WRAPPER (Sovereign Core) ===
class PyWaveCore:
    def __init__(self, dim, batch_size):
        self.dim = dim
        self.batch = batch_size
        
        # Load DLL
        if not os.path.exists(LIB_PATH):
            raise FileNotFoundError(f"Library {LIB_PATH} not found!")
            
        self.lib = ctypes.CDLL(os.path.abspath(LIB_PATH))
        
        # Signatures (v37.5 Compatible)
        self.lib.run_avx512_evolution.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float
        ]
        self.lib.update_dynamic_memory.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float
        ]
        
        # Buffers
        self.state = np.zeros((batch_size, 3, dim), dtype=np.float32)
        self.buffer = np.zeros((batch_size, 3, dim), dtype=np.float32)
        self.anchors = np.zeros(dim, dtype=np.float32)
        self.rules = np.array([-0.1, 0.8, -0.1], dtype=np.float32)
        self.grads = np.zeros(dim, dtype=np.float32) # Dummy grads

    def forward_backward(self):
        # 1. Evolution (Forward)
        self.lib.run_avx512_evolution(
            self.state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.rules.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.anchors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.batch, self.dim, 16, ctypes.c_float(0.05)
        )
        # 2. Update (Backward approximation)
        self.lib.update_dynamic_memory(
            self.anchors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.grads.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.dim, ctypes.c_float(0.01), ctypes.c_float(1.0), ctypes.c_float(0.0)
        )

# === 2. PYTORCH BASELINE (The Challenger) ===
class PyTorchBaseline(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Simulating standard Attention/Dense layer load
        # Linear layer is O(N^2) compute
        self.layer = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        
    def train_step(self, x, target):
        self.zero_grad()
        out = self.layer(x)
        out = self.act(out)
        loss = torch.mean((out - target)**2)
        loss.backward()
        # Optimizer step simulation
        with torch.no_grad():
            for p in self.parameters():
                p -= 0.01 * p.grad

# === 3. BENCHMARK ENGINE ===
def run_benchmark():
    print(f"=== PYWAVE vs PYTORCH: SCALING BENCHMARK ===")
    print(f"System: {platform.processor()} | Batch: {BATCH_SIZE}")
    print(f"Protocol: {TEST_STEPS} steps (+{WARMUP_STEPS} warmup)")
    print("-" * 75)
    print(f"{'DIMENSION':<10} | {'PyTorch (ms)':<15} | {'PyWave (ms)':<15} | {'Speedup':<10} | {'Complexity'}")
    print("-" * 75)
    
    results = []
    
    for dim in DIMENSIONS:
        # --- DATA PREP ---
        # Data generation is outside the timer
        pt_x = torch.randn(BATCH_SIZE, dim)
        pt_y = torch.randn(BATCH_SIZE, dim)
        
        # --- PYTORCH TEST ---
        model_pt = PyTorchBaseline(dim)
        # Warmup
        for _ in range(WARMUP_STEPS): model_pt.train_step(pt_x, pt_y)
        
        # Timing
        torch.set_num_threads(os.cpu_count()) # Give PyTorch full power
        start = time.perf_counter()
        for _ in range(TEST_STEPS):
            model_pt.train_step(pt_x, pt_y)
        end = time.perf_counter()
        time_pt = (end - start) * 1000 / TEST_STEPS # ms per batch
        
        # --- PYWAVE TEST ---
        model_pw = PyWaveCore(dim, BATCH_SIZE)
        # Warmup
        for _ in range(WARMUP_STEPS): model_pw.forward_backward()
        
        # Timing
        start = time.perf_counter()
        for _ in range(TEST_STEPS):
            model_pw.forward_backward()
        end = time.perf_counter()
        time_pw = (end - start) * 1000 / TEST_STEPS # ms per batch
        
        # --- METRICS ---
        speedup = time_pt / (time_pw + 1e-9)
        
        # Determine Complexity Tag
        tag = "Equal"
        if speedup > 1.5: tag = "Linear Win"
        if speedup > 5.0: tag = "DOMINATION"
        
        print(f"{dim:<10} | {time_pt:<15.2f} | {time_pw:<15.2f} | {speedup:<9.2f}x | {tag}")
        results.append({"dim": dim, "pytorch_ms": time_pt, "pywave_ms": time_pw, "speedup": speedup})

    print("-" * 75)
    
    # Save to CSV for GitHub Graphs
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["dim", "pytorch_ms", "pywave_ms", "speedup"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[SUCCESS] Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_benchmark()