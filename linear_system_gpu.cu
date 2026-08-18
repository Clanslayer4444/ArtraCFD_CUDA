

// linear_system_gpu.cu
#include <cuda_runtime.h>
#include "commons.h"

// Simple GPU kernel: copy RHS to solution (placeholder for LU/LLLU later)
__global__ void SolveLinearSystemKernel(int n, int m,
                                        const Real* __restrict__ B,
                                        Real* __restrict__ X) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < m) {
        // Dummy solver: just copy B -> X
        X[row * m + col] = B[row * m + col];
    }
}

// Host wrapper function (C-linkage so C code can call it)
extern "C" int SolveLinearSystemGPU(const int n, Real* A,
                                    const int m, Real* X, Real* B) {
    size_t bytesMat = n * n * sizeof(Real);
    size_t bytesVec = n * m * sizeof(Real);

    Real *dA, *dX, *dB;

    // Allocate device memory
    cudaMalloc((void**)&dA, bytesMat);
    cudaMalloc((void**)&dB, bytesVec);
    cudaMalloc((void**)&dX, bytesVec);

    // Copy inputs to device
    cudaMemcpy(dA, A, bytesMat, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytesVec, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    SolveLinearSystemKernel<<<grid, block>>>(n, m, dB, dX);

    // Copy result back
    cudaMemcpy(X, dX, bytesVec, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dX);

    return 0;
}



