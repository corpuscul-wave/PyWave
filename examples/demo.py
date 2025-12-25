import ctypes
import os
import sys
import time
import random
import math

# === KONFIGURATION & PHYSICS CONSTANTS ===
BATCH_SIZE = 1
DIMENSION = 64  # Размерность волнового фронта
STEPS_PER_FRAME = 10
LIB_PATH = "libpywave.dll"  # Предполагаем запуск из корня проекта

# Цвета для консоли (ANSI)
class Colors:
    RESET = "\033[0m"
    RED = "\033[31m"      # Активность (R)
    GREEN = "\033[32m"    # Память (G)
    BLUE = "\033[34m"     # Усталость (B)
    YELLOW = "\033[33m"   # Всплески

# === CTYPES WRAPPER (Low-Level Interface) ===
class PyWaveEngine:
    def __init__(self, dll_path):
        cwd = os.getcwd()
        full_path = os.path.join(cwd, dll_path)
        
        if not os.path.exists(full_path):
            print(f"CRITICAL ERROR: Kernel library not found at {full_path}")
            print("Run 'python build.py' first.")
            sys.exit(1)

        try:
            self.lib = ctypes.CDLL(full_path)
        except OSError as e:
            print(f"System Error loading DLL: {e}")
            sys.exit(1)

        # Настройка сигнатур функций (C++ signatures)
        
        # void run_avx512_evolution(float* state, float* buffer, float* rules, float* anchors, int B, int D, int steps, float noise)
        self.lib.run_avx512_evolution.argtypes = [
            ctypes.POINTER(ctypes.c_float), # state
            ctypes.POINTER(ctypes.c_float), # buffer
            ctypes.POINTER(ctypes.c_float), # rules
            ctypes.POINTER(ctypes.c_float), # anchors
            ctypes.c_int,                   # B
            ctypes.c_int,                   # D
            ctypes.c_int,                   # steps
            ctypes.c_float                  # noise_level
        ]

        # void update_dynamic_memory(...)
        self.lib.update_dynamic_memory.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float
        ]

        print(f"[PyWave] Engine attached. AVX512/OMP ready. D={DIMENSION}")

    def create_buffer(self, size):
        return (ctypes.c_float * size)()

# === VISUALIZATION TOOLS ===
def render_wave(state_ptr, dim):
    """Рисует ASCII тепловую карту текущего состояния (R-канал)"""
    # Считываем R-канал (первые D элементов)
    chars = " .:-=+*#%@"
    output = []
    
    # Получаем срез памяти
    r_channel = state_ptr[:dim]
    b_channel = state_ptr[2*dim:3*dim] # Fatigue channel
    
    line = ""
    for i in range(dim):
        val = r_channel[i]
        fatigue = b_channel[i]
        
        # Индекс яркости
        idx = int(max(0, min(len(chars) - 1, (val + 2.0) * 2)))
        char = chars[idx]
        
        # Color logic based on state
        if fatigue > 0.8: # Gating active
            line += f"{Colors.BLUE}{char}{Colors.RESET}"
        elif val > 1.0:   # High excitation
            line += f"{Colors.RED}{char}{Colors.RESET}"
        elif val < -1.0:  # Inhibition
            line += f"{Colors.GREEN}{char}{Colors.RESET}"
        else:
            line += char
            
    return f"[{line}] Avg E: {sum(map(abs, r_channel))/dim:.4f}"

# === MAIN SIMULATION LOOP ===
def main():
    engine = PyWaveEngine(LIB_PATH)

    # 1. Memory Allocation (Pinned Memory emulation)
    total_size = BATCH_SIZE * 3 * DIMENSION
    state = engine.create_buffer(total_size)
    buffer = engine.create_buffer(total_size)
    
    # 2. Initialize Rules (Reaction-Diffusion parameters)
    # [Left, Self, Right] weights
    rules = (ctypes.c_float * 3)(0.5, -0.1, 0.5) 
    
    # 3. Initialize Anchors (Long-term memory)
    anchors = engine.create_buffer(DIMENSION)
    
    # Инициализация случайным шумом (Start condition)
    for i in range(total_size):
        state[i] = (random.random() - 0.5) * 0.1

    print("\n=== STARTING PYWAVE RESONANCE DEMO ===")
    print("Watching for: Phase Interference (Red) & Entropy Gating (Blue lock)\n")
    
    try:
        t = 0
        while True:
            # Динамический шум: синусоида для проверки реакции системы
            current_noise = 0.5 * (1.0 + math.sin(t * 0.1))
            
            # Взрыв энтропии каждые 50 кадров (тест устойчивости)
            if t % 50 == 0: 
                print(f"{Colors.YELLOW}>>> ENTROPY INJECTION <<<{Colors.RESET}")
                current_noise = 5.0 

            # === CORE EVOLUTION STEP ===
            engine.lib.run_avx512_evolution(
                state, buffer, rules, anchors, 
                BATCH_SIZE, DIMENSION, STEPS_PER_FRAME, 
                ctypes.c_float(current_noise)
            )

            # Визуализация
            vis = render_wave(state, DIMENSION)
            print(f"Step {t:04d} | Noise: {current_noise:.2f} | {vis}")
            
            time.sleep(0.05)
            t += 1

    except KeyboardInterrupt:
        print("\n[STOP] Simulation halted by user.")

if __name__ == "__main__":
    main()
