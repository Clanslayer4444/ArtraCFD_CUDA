#include <cuda_runtime.h>
#include <math.h>
#include "commons.h"
#include "gpu_eigen.h"
#include "gpu_eigen_device.h"

/* 
   Device Helper: Eigenvalue Splitting
   Matches cfd_commons.c logic: P = 0.5(L + |L|), N = 0.5(L - |L|)
   - */
__device__ __forceinline__
void EigenvalueSplitting_dev(
    int splitter,
    const Real *Lambda,
    Real *LambdaP,
    Real *LambdaN)
{
    // 0 = LF (Lax-Friedrichs) - often handled differently, but here we do basic split
    // 1 = Steger-Warming
    for (int k = 0; k < DIMU; ++k) {
        Real val = Lambda[k];
        Real absVal = fabs(val);
        LambdaP[k] = 0.5 * (val + absVal);
        LambdaN[k] = 0.5 * (val - absVal);
    }
}

/* 
   Kernel: Eigenvalue Test
   */
__global__
void EigenTest_kernel(
    int s, 
    Real gamma, 
    const Real *Uo, 
    Real *Lambda, 
    Real *L, 
    Real *R_mat
)
{
    int tid = threadIdx.x;
    if (tid > 0) return; // Single thread test

    Eigenvalue_dev(s, Uo, Lambda);
    EigenvectorL_dev(s, gamma, Uo, L); // L is flat [DIMU*DIMU]
    EigenvectorR_dev(s, Uo, R_mat);    // R is flat [DIMU*DIMU]
}

/* 
   Kernel: Eigenvalue Splitting Test
    */
__global__
void EigenSplitTest_kernel(
    int splitter,
    const Real *Lambda,
    Real *LambdaP,
    Real *LambdaN
)
{
    int tid = threadIdx.x;
    if (tid > 0) return;

    EigenvalueSplitting_dev(splitter, Lambda, LambdaP, LambdaN);
}

/* 
   Host Launcher: Eigen Test
   */
extern "C"
int LaunchEigenTestGPU(
    int N,
    int s,
    Real gamma,
    const Real *Uo_h,
    Real *Lambda_h,
    Real *L_h,
    Real *R_h
)
{
    Real *d_Uo, *d_Lambda, *d_L, *d_R;
    
    cudaMalloc(&d_Uo, DIMUo * sizeof(Real));
    cudaMalloc(&d_Lambda, DIMU * sizeof(Real));
    cudaMalloc(&d_L, DIMU * DIMU * sizeof(Real));
    cudaMalloc(&d_R, DIMU * DIMU * sizeof(Real));

    cudaMemcpy(d_Uo, Uo_h, DIMUo * sizeof(Real), cudaMemcpyHostToDevice);

    EigenTest_kernel<<<1, 1>>>(s, gamma, d_Uo, d_Lambda, d_L, d_R);

    cudaMemcpy(Lambda_h, d_Lambda, DIMU * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(L_h, d_L, DIMU * DIMU * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(R_h, d_R, DIMU * DIMU * sizeof(Real), cudaMemcpyDeviceToHost);

    cudaFree(d_Uo); cudaFree(d_Lambda); cudaFree(d_L); cudaFree(d_R);
    return 0;
}


extern "C"
int LaunchEigenSplitTestGPU(
    int splitter,
    const Real *Lambda,
    Real *LambdaP,
    Real *LambdaN
)
{
    Real *d_Lambda, *d_LambdaP, *d_LambdaN;
    size_t sz = DIMU * sizeof(Real);

    cudaMalloc(&d_Lambda, sz);
    cudaMalloc(&d_LambdaP, sz);
    cudaMalloc(&d_LambdaN, sz);

    cudaMemcpy(d_Lambda, Lambda, sz, cudaMemcpyHostToDevice);

    EigenSplitTest_kernel<<<1, 1>>>(splitter, d_Lambda, d_LambdaP, d_LambdaN);

    cudaMemcpy(LambdaP, d_LambdaP, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(LambdaN, d_LambdaN, sz, cudaMemcpyDeviceToHost);

    cudaFree(d_Lambda); cudaFree(d_LambdaP); cudaFree(d_LambdaN);
    return 0;
}