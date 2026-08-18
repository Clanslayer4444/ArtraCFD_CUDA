import os
import numpy as np
import sys

# List of extensions that contain actual numbers to compare
# We skip .case, .did, .geo as they are often headers/metadata
EXTENSIONS_TO_CHECK = ['.rho', '.u', '.v', '.w', '.p', '.T']

def read_file(filepath):
    """Reads a numeric file into a numpy array."""
    try:
        # Try reading as simple text list of numbers
        return np.loadtxt(filepath)
    except Exception as e:
        # If read fails (e.g. it's binary or weird header), we skip it safely
        return None

def compare(folder_cpu, folder_gpu):
    print(f"\n🚀 Starting Comparison: {folder_cpu} vs {folder_gpu}")
    
    # Get list of relevant files from CPU folder
    files = sorted([f for f in os.listdir(folder_cpu) 
                    if any(f.endswith(ext) for ext in EXTENSIONS_TO_CHECK)])
    
    if not files:
        print(f"❌ No field files (rho, u, v...) found in {folder_cpu}.")
        return

    mismatch_count = 0
    passed_count = 0

    print(f"   Found {len(files)} files to verify...")

    for f in files:
        path_cpu = os.path.join(folder_cpu, f)
        path_gpu = os.path.join(folder_gpu, f)
        
        if not os.path.exists(path_gpu):
            print(f"[MISSING] {f} exists in CPU but NOT in GPU folder.")
            mismatch_count += 1
            continue

        data_cpu = read_file(path_cpu)
        data_gpu = read_file(path_gpu)
        
        if data_cpu is None or data_gpu is None:
            print(f"[SKIP] Could not read data from {f} (Format issue?)")
            continue

        # Check if array shapes match
        if data_cpu.shape != data_gpu.shape:
            print(f"[FAIL] {f}: Shape mismatch! CPU={data_cpu.shape}, GPU={data_gpu.shape}")
            mismatch_count += 1
            continue

        # Calculate Error
        diff = np.abs(data_cpu - data_gpu)
        max_diff = np.max(diff)

        # Tolerance: 1e-4 is safe for mixed precision, 1e-10 for double precision
        if max_diff > 1e-4:
            print(f"❌ [FAIL] {f}: Max Diff = {max_diff:.6f}")
            mismatch_count += 1
        else:
            # Uncomment line below if you want to see every PASS (it might be spammy)
            # print(f"✅ [PASS] {f}: Max Diff = {max_diff:.2e}")
            passed_count += 1

    print("-" * 50)
    if mismatch_count == 0:
        print(f"🎉 SUCCESS! All {passed_count} files match perfectly.")
        print("   Your GPU solver is mathematically identical to the CPU version.")
    else:
        print(f"💀 FAILURE! Found {mismatch_count} files with mismatches.")

if __name__ == "__main__":
    # Ensure these folder names match exactly what you have
    cpu_dir = "data_cpu"
    gpu_dir = "data_gpu"
    
    if not os.path.exists(cpu_dir) or not os.path.exists(gpu_dir):
        print(f"Error: Folders '{cpu_dir}' or '{gpu_dir}' not found.")
        print("Did you rename 'data' to 'data_cpu' and 'data_gpu'?")
    else:
        compare(cpu_dir, gpu_dir)