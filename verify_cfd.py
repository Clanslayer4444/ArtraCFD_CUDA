import numpy as np
import sys
import os

def verify_output(baseline_file, test_file, tolerance=1e-6):
    if not os.path.exists(baseline_file) or not os.path.exists(test_file):
        print(f"Error: One of the files not found: {baseline_file}, {test_file}")
        sys.exit(1)

    # Assumes binary file of floats (change dtype if using double)
    # If data is ASCII, change to np.loadtxt
    try:
        data_base = np.fromfile(baseline_file, dtype=np.float32)
        data_test = np.fromfile(test_file, dtype=np.float32)
    except Exception as e:
        print(f"Error reading files: {e}")
        sys.exit(1)

    if data_base.shape != data_test.shape:
        print(f"CRITICAL: Shape mismatch! Baseline: {data_base.shape}, Test: {data_test.shape}")
        sys.exit(1)

    # Calculate metrics
    diff = np.abs(data_base - data_test)
    max_err = np.max(diff)
    l2_norm = np.linalg.norm(data_base - data_test) / np.linalg.norm(data_base)

    print(f"--- Verification Results ---")
    print(f"Max Absolute Error: {max_err:.8e}")
    print(f"Relative L2 Error:  {l2_norm:.8e}")

    if max_err > tolerance:
        print("RESULT: FAILED - Divergence detected beyond tolerance.")
        sys.exit(1)
    else:
        print("RESULT: PASSED - Matches baseline.")
        sys.exit(0)

if __name__ == "__main__":
    # Usage: python verify_cfd.py baseline.bin current.bin
    if len(sys.argv) < 3:
        print("Usage: python verify_cfd.py <baseline_file> <test_file>")
    else:
        verify_output(sys.argv[1], sys.argv[2])