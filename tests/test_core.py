import pytest
import numpy as np
import os
import sys
import ctypes

# Добавляем корневую папку в путь, чтобы видеть pywave_api
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pywave_api import PyWave # Предполагаем, что ваш API файл так называется

LIB_PATH = "libpywave.dll" if os.name == 'nt' else "libpywave.so"

@pytest.fixture
def engine():
    if not os.path.exists(LIB_PATH):
        pytest.skip("DLL not compiled. Run 'python build.py' first.")
    return PyWave(dim=128, batch_size=4, lib_path=LIB_PATH)

def test_initialization(engine):
    assert engine.state.shape == (4, 3, 128)
    assert np.all(engine.state == 0)

def test_evolution_integrity(engine):
    # Тест на отсутствие NaN и взрывов
    anchors = np.random.randn(128).astype(np.float32)
    rules = np.array([-0.1, 0.8, -0.1], dtype=np.float32)
    
    engine.evolve(anchors, rules, steps=16)
    
    assert not np.isnan(engine.state).any()
    assert not np.isinf(engine.state).any()
    # Проверка диапазона (Clamping test)
    assert np.max(engine.state) <= 5.0
    assert np.min(engine.state) >= -5.0

def test_memory_update(engine):
    # Тест ADS
    anchors = np.ones(128, dtype=np.float32)
    grads = np.ones(128, dtype=np.float32) * 0.1
    
    # Симуляция "Паники" -> веса должны уменьшиться (Decay)
    engine.lib.update_dynamic_memory(
        anchors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        grads.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        128, ctypes.c_float(0.01), ctypes.c_float(0.0), ctypes.c_float(1.0) # High panic
    )
    
    assert np.mean(anchors) < 1.0 # Должны уменьшиться из-за Decay