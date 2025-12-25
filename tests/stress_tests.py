import numpy as np
import sys
import os
import time

# Добавляем корневую директорию в путь, чтобы видеть pywave_api
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pywave_api import PyWave

def run_stress_suite():
    print("=== PyWave v37.5 SOVEREIGN STRESS TEST SUITE ===")
    
    # Инициализация
    try:
        dim = 2048
        batch_size = 32
        engine = PyWave(dim=dim, batch_size=batch_size)
        print(f"[INIT] Engine loaded. Library found.")
    except FileNotFoundError as e:
        print(f"[FATAL] {e}")
        print("Please compile the DLL using 'python build.py' first.")
        return

    # 1. ТЕСТ: ВЗРЫВНОЙ СИГНАЛ (Exploding Input)
    print("\n[1/4] Testing Overload Stability...")
    huge_anchors = np.random.randn(dim).astype(np.float32) * 1000.0
    rules = np.array([5.0, -10.0, 5.0], dtype=np.float32)
    
    # 100 шагов с огромным входом
    engine.evolve(huge_anchors, rules, steps=100, noise=1.0)
    
    max_val = np.max(np.abs(engine.state))
    # Лимит в C++ коде MAX_AMPLITUDE = 5.0f
    if max_val <= 5.001: 
        print(f"  PASS: Clamping holds. Max amplitude: {max_val:.4f}")
    else:
        print(f"  FAIL: System exploded! Max amplitude: {max_val}")

    # 2. ТЕСТ: ДЛИТЕЛЬНАЯ ЭНТРОПИЯ (10k Steps)
    print("\n[2/4] Testing Long-term Entropy (10,000 steps)...")
    start_time = time.time()
    # Прогоняем 100 циклов по 100 шагов
    for i in range(100):
        engine.evolve(huge_anchors, rules, steps=100, noise=0.1)
    
    if not np.isnan(engine.state).any():
        print(f"  PASS: No NaNs after 10k steps. Time: {time.time()-start_time:.2f}s")
    else:
        print("  FAIL: NaNs detected in long-term evolution.")

    # 3. ТЕСТ: ZEN DECAY (Затухание при тишине)
    print("\n[3/4] Testing Zen Decay (Silence test)...")
    zero_anchors = np.zeros(dim, dtype=np.float32)
    # Искусственно "зажигаем" все нейроны до 3.0
    engine.state.fill(3.0) 
    
    # Запускаем эволюцию без стимулов (noise=0, anchors=0)
    for _ in range(50):
        engine.evolve(zero_anchors, rules, steps=10, noise=0.0)
    
    mean_activity = np.mean(np.abs(engine.state))
    print(f"  Result: Activity dropped from 3.0 to {mean_activity:.6f}")
    
    if mean_activity < 1.0:
        print("  PASS: Zen Decay successfully suppresses residual activity.")
    else:
        print("  FAIL: System did not decay enough.")

    # 4. ТЕСТ: ИЗОЛЯЦИЯ БАТЧЕЙ (Batch Isolation)
    print("\n[4/4] Testing Batch Isolation...")
    engine.state.fill(0)
    # Возбуждаем ТОЛЬКО 0-й батч
    engine.state[0, :, :] = 2.0 
    
    engine.evolve(zero_anchors, rules, steps=20, noise=0.0)
    
    # Проверяем сумму активности во всех батчах, кроме 0-го
    leaks = np.sum(np.abs(engine.state[1:]))
    if leaks == 0:
        print("  PASS: Batch threads are perfectly isolated.")
    else:
        print(f"  FAIL: Memory leak between batches! Sum: {leaks}")

if __name__ == "__main__":
    run_stress_suite()
