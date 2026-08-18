import os
import numpy as np

# We assume the solver uses 'double' (float64) for Real numbers
DTYPE = np.float64 

def compare(folder_cpu, folder_gpu):
    print(f"\n⚡ Starting BINARY Comparison: {folder_cpu} vs {folder_gpu}")
    
    # Extensions for field data
    exts = ['.rho', '.u', '.v', '.w', '.p', '.T']
    files = sorted([f for f in os.listdir(folder_cpu) if any(f.endswith(e) for e in exts)])
    
    if not files:
        print("❌ No data files found.")
        return

    mismatch_count = 0

    for f in files:
        p_cpu = os.path.join(folder_cpu, f)
        p_gpu = os.path.join(folder_gpu, f)
        
        if not os.path.exists(p_gpu):
            continue

        # Read entire file as a raw binary array of floats
        # The text header will turn into "garbage" numbers, but they will be
        # IDENTICAL garbage on both CPU and GPU, so they will cancel out.
        try:
            d_cpu = np.fromfile(p_cpu, dtype=DTYPE)
            d_gpu = np.fromfile(p_gpu, dtype=DTYPE)
        except Exception as e:
            print(f"[SKIP] {f} could not be read: {e}")
            continue

        # If sizes differ, the file is definitely wrong
        if d_cpu.size != d_gpu.size:
            print(f"❌ [FAIL] {f}: File size mismatch!")
            mismatch_count += 1
            continue

        # Compare Data
        diff = np.abs(d_cpu - d_gpu)
        max_diff = np.max(diff)

        # We ignore NaNs (which might happen if header text is read as floats)
        max_diff = np.nanmax(diff)

        if max_diff > 1e-10:
            print(f"❌ [FAIL] {f}: Max Diff = {max_diff:.6e}")
            mismatch_count += 1
        else:
            print(f"✅ [PASS] {f}: Matches perfectly.")

    print("-" * 40)
    if mismatch_count == 0:
        print("🎉 SUCCESS: CPU and GPU results are bit-wise identical.")
    else:
        print(f"💀 FAILURE: Found {mismatch_count} mismatches.")

if __name__ == "__main__":
    compare("data_cpu", "data_gpu")