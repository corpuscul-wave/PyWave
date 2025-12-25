import os
import platform
import subprocess
import sys

def build():
    system = platform.system()
    print(f"[PyWave Builder] Detected system: {system}")
    
    # Определение компилятора и флагов
    compiler = "g++"
    source = "core/pywave_engine.cpp"
    
    if system == "Windows":
        output = "libpywave.dll"
        # Windows flags (Static linking needed for portability)
        flags = [
            "-O3", "-shared", "-fopenmp", "-mavx2", "-mfma", 
            "-static", "-static-libgcc", "-static-libstdc++"
        ]
    else:
        output = "libpywave.so"
        # Linux/Mac flags (-fPIC is crucial)
        flags = [
            "-O3", "-shared", "-fopenmp", "-mavx2", "-mfma", "-fPIC"
        ]

    cmd = [compiler] + flags + [source, "-o", output]
    
    print(f"[Building] {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print(f"[Success] Compiled to {output}")
        print("You can now run 'python examples/demo.py' or benchmarks.")
    except subprocess.CalledProcessError:
        print("[Error] Compilation failed.")
        print("Ensure you have MinGW-w64 (Windows) or build-essential (Linux) installed.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"[Error] Compiler '{compiler}' not found in PATH.")
        sys.exit(1)

if __name__ == "__main__":
    if not os.path.exists("core"):
        os.makedirs("core", exist_ok=True)
        print("Please place 'pywave_engine.cpp' inside 'core/' folder!")
    else:
        build()