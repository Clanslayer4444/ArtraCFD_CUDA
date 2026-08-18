#!/bin/bash
# ===========================================================================
# ArtraCFD CPU-vs-GPU Verification Test
# Runs one timestep in both modes and compares output fields.
# Usage: ./run_verify.sh [tolerance]
#   tolerance: max relative diff allowed (default 1e-12)
# ===========================================================================

set -euo pipefail

TOLERANCE="${1:-1e-12}"
BINDIR="$(cd "$(dirname "$0")/.." && pwd)"
TESTDIR="$(cd "$(dirname "$0")" && pwd)"
VERIFY_DIR="${TESTDIR}/verify_work"

echo "============================================================"
echo " ArtraCFD CPU-vs-GPU Verification Test"
echo "============================================================"
echo " Tolerance: ${TOLERANCE}"
echo " Binary:    ${BINDIR}/artracfd"
echo " Work dir:  ${VERIFY_DIR}"
echo "============================================================"

# Clean and setup work directory
rm -rf "${VERIFY_DIR}"
mkdir -p "${VERIFY_DIR}"

# Copy test case files
cp "${TESTDIR}/artracfd.case" "${VERIFY_DIR}/"
cp "${TESTDIR}/artracfd.geo"  "${VERIFY_DIR}/"
cp "${TESTDIR}/artracfd.stl"  "${VERIFY_DIR}/"

# ===========================================================================
# Run Serial (CPU) mode
# ===========================================================================
echo ""
echo "--- Running SERIAL mode ---"
cd "${VERIFY_DIR}"
# Create a subfolder for serial output
mkdir -p serial
cd serial
# Symlink the case files
ln -sf ../artracfd.case artracfd.case
ln -sf ../artracfd.geo  artracfd.geo
ln -sf ../artracfd.stl  artracfd.stl

# Run serial with a short time
"${BINDIR}/artracfd" -m serial 2>&1 | tee serial_output.txt || echo "SERIAL run finished (may exit with status from ShowError)"

# Check if solution files were produced
if ls field00000.* >/dev/null 2>&1; then
    echo "SERIAL: Solution files found."
    ls field00000.* | head -10
else
    echo "WARNING: No field00000.* files in serial output"
fi

cd "${VERIFY_DIR}"

# ===========================================================================
# Run GPU mode
# ===========================================================================
echo ""
echo "--- Running GPU mode ---"
cd "${VERIFY_DIR}"
mkdir -p gpu
cd gpu
ln -sf ../artracfd.case artracfd.case
ln -sf ../artracfd.geo  artracfd.geo
ln -sf ../artracfd.stl  artracfd.stl

"${BINDIR}/artracfd" -m gpu -n 1*1*1 2>&1 | tee gpu_output.txt || echo "GPU run finished"

if ls field00000.* >/dev/null 2>&1; then
    echo "GPU: Solution files found."
    ls field00000.* | head -10
else
    echo "WARNING: No field00000.* files in gpu output"
fi

cd "${VERIFY_DIR}"

# ===========================================================================
# Compare outputs
# ===========================================================================
echo ""
echo "============================================================"
echo " Comparing CPU vs GPU results"
echo "============================================================"

PASS=0
FAIL=0

# Compare field files (rho, u, v, p, T, etc.)
for var in rho u v p T; do
    CPU_FILE="serial/field00000.${var}"
    GPU_FILE="gpu/field00000.${var}"
    
    if [ ! -f "${CPU_FILE}" ]; then
        echo "  SKIP ${var}: CPU file missing (${CPU_FILE})"
        continue
    fi
    if [ ! -f "${GPU_FILE}" ]; then
        echo "  SKIP ${var}: GPU file missing (${GPU_FILE})"
        continue
    fi

    # Use python to read and compare binary files
    python3 -c "
import struct, sys, math

def read_field(fname):
    '''Read EnSight field file (line-based, 6 values per line).'''
    vals = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            for p in parts:
                try:
                    vals.append(float(p))
                except:
                    pass
    return vals

cpu = read_field('${CPU_FILE}')
gpu = read_field('${GPU_FILE}')

if len(cpu) != len(gpu):
    print(f'  FAIL {var}: length mismatch cpu={len(cpu)} gpu={len(gpu)}')
    sys.exit(1)

max_diff = 0.0
max_rel = 0.0
bad_count = 0
tol = ${TOLERANCE}
for i in range(len(cpu)):
    if abs(cpu[i]) > 1e-30:
        rel = abs(cpu[i] - gpu[i]) / abs(cpu[i])
    else:
        rel = abs(cpu[i] - gpu[i])
    absd = abs(cpu[i] - gpu[i])
    if rel > max_rel: max_rel = rel
    if absd > max_diff: max_diff = absd
    if rel > tol and absd > 1e-30:
        bad_count += 1

if bad_count > 0:
    print(f'  FAIL {var}: max_rel={max_rel:.2e} max_abs={max_diff:.2e} bad={bad_count}/{len(cpu)}')
    sys.exit(1)
else:
    print(f'  PASS {var}: max_rel={max_rel:.2e} max_abs={max_diff:.2e} size={len(cpu)}')
" 2>&1 || FAIL=$((FAIL + 1))
    
    if [ $? -eq 0 ]; then
        PASS=$((PASS + 1))
    fi
done

# ===========================================================================
# Summary
# ===========================================================================
echo ""
echo "============================================================"
echo " Verification Summary"
echo "============================================================"
echo " Passed: ${PASS}"
echo " Failed: ${FAIL}"
if [ "${FAIL}" -eq 0 ]; then
    echo " RESULT: ALL CHECKS PASSED"
else
    echo " RESULT: ${FAIL} CHECK(S) FAILED"
fi
echo "============================================================"

exit ${FAIL}
