#include <cuda_runtime.h>
#include <stdio.h>
#include "commons.h"
#include "gpu_convective_flux.h"

extern cudaStream_t computeStream;

#define TPB 256
#define restrict __restrict__


#define WENO_SL -2
#define WENO_SR 3
#define WENO_SCHEME 1    /* 1 = WENO5 */
#define DIMU 5
#define DIMUo 6

__device__ inline Real Square(const Real x) { return x * x; }
__device__ inline int IdxNode(const int k, const int j, const int i, const int ny, const int nx) {
    return (k * ny + j) * nx + i;
}

/* ==========================================
   1:1 PORT OF WENO3 & WENO5 RECONSTRUCTION
   ==========================================*/
__device__ void WENO3_dev(Real F[][DIMU], Real Fhat[]) {
    Real omega[2], q[2], IS[2], alpha[2];
    const Real C[2] = {1.0 / 3.0, 2.0 / 3.0};
    const Real epsilon = 1.0e-6;

    for (int r = 0; r < DIMU; ++r) {
        IS[0] = Square(F[1][r] - F[0][r]);
        IS[1] = Square(F[2][r] - F[1][r]);

        alpha[0] = C[0] / Square(epsilon + IS[0]);
        alpha[1] = C[1] / Square(epsilon + IS[1]);

        Real sum_alpha = alpha[0] + alpha[1];
        omega[0] = alpha[0] / sum_alpha;
        omega[1] = alpha[1] / sum_alpha;

        q[0] = (1.0 / 2.0) * (-F[0][r] + 3.0 * F[1][r]);
        q[1] = (1.0 / 2.0) * (F[1][r] + F[2][r]);

        Fhat[r] = omega[0] * q[0] + omega[1] * q[1];
    }
}

__device__ void WENO5_dev(Real F[][DIMU], Real Fhat[]) {
    Real omega[3], q[3], IS[3], alpha[3];
    const Real C[3] = {0.1, 0.6, 0.3};
    const Real epsilon = 1.0e-6;

    for (int r = 0; r < DIMU; ++r) {
        Real f0 = F[0][r], f1 = F[1][r], f2 = F[2][r], f3 = F[3][r], f4 = F[4][r];

        IS[0] = (13.0/12.0)*Square(f0 - 2.0*f1 + f2) + 0.25*Square(f0 - 4.0*f1 + 3.0*f2);
        IS[1] = (13.0/12.0)*Square(f1 - 2.0*f2 + f3) + 0.25*Square(f1 - f3);
        IS[2] = (13.0/12.0)*Square(f2 - 2.0*f3 + f4) + 0.25*Square(3.0*f2 - 4.0*f3 + f4);

        alpha[0] = C[0] / Square(epsilon + IS[0]);
        alpha[1] = C[1] / Square(epsilon + IS[1]);
        alpha[2] = C[2] / Square(epsilon + IS[2]);

        Real sum = alpha[0] + alpha[1] + alpha[2];
        omega[0] = alpha[0]/sum; omega[1] = alpha[1]/sum; omega[2] = alpha[2]/sum;

        q[0] = (1.0/6.0)*(2.0*f0 - 7.0*f1 + 11.0*f2);
        q[1] = (1.0/6.0)*(-f1 + 5.0*f2 + 2.0*f3);
        q[2] = (1.0/6.0)*(2.0*f2 + 5.0*f3 - f4);

        Fhat[r] = omega[0]*q[0] + omega[1]*q[1] + omega[2]*q[2];
    }
}


/* =========================================================================
    FLUX SPLITTING MATH
   ========================================================================= */
__device__ void SymmetricAverage_dev(const Real gamma, const Real UL[], const Real UR[], Real Uo[]) {
    const Real rhoL = UL[0], uL = UL[1]/UL[0], vL = UL[2]/UL[0], wL = UL[3]/UL[0];
    const Real hTL = (UL[4]/UL[0])*gamma - 0.5*(uL*uL + vL*vL + wL*wL)*(gamma - 1.0);
    const Real rhoR = UR[0], uR = UR[1]/UR[0], vR = UR[2]/UR[0], wR = UR[3]/UR[0];
    const Real hTR = (UR[4]/UR[0])*gamma - 0.5*(uR*uR + vR*vR + wR*wR)*(gamma - 1.0);

    
    Uo[0] = 0.0;
    Uo[1] = 0.5 * (uL + uR);
    Uo[2] = 0.5 * (vL + vR);
    Uo[3] = 0.5 * (wL + wR);
    Uo[4] = 0.5 * (hTL + hTR);
    Uo[5] = sqrt((gamma - 1.0) * (Uo[4] - 0.5 * (Uo[1]*Uo[1] + Uo[2]*Uo[2] + Uo[3]*Uo[3])));
}


__device__ void Eigenvalue_dev(const int s, const Real Uo[], Real Lambda[]) {
    Lambda[0] = Uo[s+1] - Uo[5];
    Lambda[1] = Uo[s+1]; Lambda[2] = Uo[s+1]; Lambda[3] = Uo[s+1];
    Lambda[4] = Uo[s+1] + Uo[5];
}

__device__ void StegerWarming_dev(const Real Lambda[], Real LambdaP[], Real LambdaN[]) {    
    const Real epsilon = 1.0e-3; // The CPU's exact entropy fix
    for (int r = 0; r < DIMU; ++r) {
        Real smoothed_abs = sqrt(Lambda[r] * Lambda[r] + epsilon * epsilon);
        LambdaP[r] = 0.5 * (Lambda[r] + smoothed_abs);
        LambdaN[r] = 0.5 * (Lambda[r] - smoothed_abs);
    }
}

__device__ void LocalLaxFriedrichs_dev(const Real Lambda[], Real LambdaP[], Real LambdaN[]) {
    // This exact formulation prevents microscopic floating-point deviations
    const Real lambdaStar = fabs(Lambda[2]) + Lambda[4] - Lambda[2]; 
    
    for (int r = 0; r < DIMU; ++r) {
        LambdaP[r] = 0.5 * (Lambda[r] + lambdaStar);
        LambdaN[r] = 0.5 * (Lambda[r] - lambdaStar);
    }
}

__device__ void EigenvectorL_dev(const int s, const Real gamma, const Real Uo[], Real L[][DIMU]) {
    const Real u = Uo[1], v = Uo[2], w = Uo[3], c = Uo[5];
    const Real q = 0.5 * (u*u + v*v + w*w);
    const Real b = (gamma - 1.0) / (2.0 * c * c);
    const Real d = 1.0 / (2.0 * c);

    if (s == 0) {
        L[0][0] = b*q + d*u;       L[0][1] = -b*u - d;      L[0][2] = -b*v;          L[0][3] = -b*w;          L[0][4] = b;
        L[1][0] = -2.0*b*q + 1.0;  L[1][1] = 2.0*b*u;       L[1][2] = 2.0*b*v;       L[1][3] = 2.0*b*w;       L[1][4] = -2.0*b;
        L[2][0] = -2.0*b*q*v;      L[2][1] = 2.0*b*v*u;     L[2][2] = 2.0*b*v*v+1.0; L[2][3] = 2.0*b*w*v;     L[2][4] = -2.0*b*v;
        L[3][0] = -2.0*b*q*w;      L[3][1] = 2.0*b*w*u;     L[3][2] = 2.0*b*w*v;     L[3][3] = 2.0*b*w*w+1.0; L[3][4] = -2.0*b*w;
        L[4][0] = b*q - d*u;       L[4][1] = -b*u + d;      L[4][2] = -b*v;          L[4][3] = -b*w;          L[4][4] = b;
    } else if (s == 1) {
        L[0][0] = b*q + d*v;       L[0][1] = -b*u;          L[0][2] = -b*v - d;      L[0][3] = -b*w;          L[0][4] = b;
        L[1][0] = -2.0*b*q*u;      L[1][1] = 2.0*b*u*u+1.0; L[1][2] = 2.0*b*v*u;     L[1][3] = 2.0*b*w*u;     L[1][4] = -2.0*b*u;
        L[2][0] = -2.0*b*q + 1.0;  L[2][1] = 2.0*b*u;       L[2][2] = 2.0*b*v;       L[2][3] = 2.0*b*w;       L[2][4] = -2.0*b;
        L[3][0] = -2.0*b*q*w;      L[3][1] = 2.0*b*w*u;     L[3][2] = 2.0*b*w*v;     L[3][3] = 2.0*b*w*w+1.0; L[3][4] = -2.0*b*w;
        L[4][0] = b*q - d*v;       L[4][1] = -b*u;          L[4][2] = -b*v + d;      L[4][3] = -b*w;          L[4][4] = b;
    } else {
        L[0][0] = b*q + d*w;       L[0][1] = -b*u;          L[0][2] = -b*v;          L[0][3] = -b*w - d;      L[0][4] = b;
        L[1][0] = -2.0*b*q*u;      L[1][1] = 2.0*b*u*u+1.0; L[1][2] = 2.0*b*v*u;     L[1][3] = 2.0*b*w*u;     L[1][4] = -2.0*b*u;
        L[2][0] = -2.0*b*q*v;      L[2][1] = 2.0*b*v*u;     L[2][2] = 2.0*b*v*v+1.0; L[2][3] = 2.0*b*w*v;     L[2][4] = -2.0*b*v;
        L[3][0] = -2.0*b*q + 1.0;  L[3][1] = 2.0*b*u;       L[3][2] = 2.0*b*v;       L[3][3] = 2.0*b*w;       L[3][4] = -2.0*b;
        L[4][0] = b*q - d*w;       L[4][1] = -b*u;          L[4][2] = -b*v;          L[4][3] = -b*w + d;      L[4][4] = b;
    }
}

__device__ void EigenvectorR_dev(const int s, const Real Uo[], Real R[][DIMU]) {
    const Real u = Uo[1], v = Uo[2], w = Uo[3], hT = Uo[4], c = Uo[5];
    const Real q = 0.5 * (u*u + v*v + w*w);
    if (s == 0) {
        R[0][0] = 1.0;         R[0][1] = 1.0;        R[0][2] = 0.0;  R[0][3] = 0.0;  R[0][4] = 1.0;
        R[1][0] = u - c;       R[1][1] = u;          R[1][2] = 0.0;  R[1][3] = 0.0;  R[1][4] = u + c;
        R[2][0] = v;           R[2][1] = 0.0;        R[2][2] = 1.0;  R[2][3] = 0.0;  R[2][4] = v;
        R[3][0] = w;           R[3][1] = 0.0;        R[3][2] = 0.0;  R[3][3] = 1.0;  R[3][4] = w;
        R[4][0] = hT - u*c;    R[4][1] = u*u - q;    R[4][2] = v;    R[4][3] = w;    R[4][4] = hT + u*c;
    } else if (s == 1) {
        R[0][0] = 1.0;         R[0][1] = 0.0;  R[0][2] = 1.0;        R[0][3] = 0.0;  R[0][4] = 1.0;
        R[1][0] = u;           R[1][1] = 1.0;  R[1][2] = 0.0;        R[1][3] = 0.0;  R[1][4] = u;
        R[2][0] = v - c;       R[2][1] = 0.0;  R[2][2] = v;          R[2][3] = 0.0;  R[2][4] = v + c;
        R[3][0] = w;           R[3][1] = 0.0;  R[3][2] = 0.0;        R[3][3] = 1.0;  R[3][4] = w;
        R[4][0] = hT - v*c;    R[4][1] = u;    R[4][2] = v*v - q;    R[4][3] = w;    R[4][4] = hT + v*c;
    } else {
        R[0][0] = 1.0;         R[0][1] = 0.0;  R[0][2] = 0.0;  R[0][3] = 1.0;        R[0][4] = 1.0;
        R[1][0] = u;           R[1][1] = 1.0;  R[1][2] = 0.0;  R[1][3] = 0.0;        R[1][4] = u;
        R[2][0] = v;           R[2][1] = 0.0;  R[2][2] = 1.0;  R[2][3] = 0.0;        R[2][4] = v;
        R[3][0] = w - c;       R[3][1] = 0.0;  R[3][2] = 0.0;  R[3][3] = w;          R[3][4] = w + c;
        R[4][0] = hT - w*c;    R[4][1] = u;    R[4][2] = v;    R[4][3] = w*w - q;    R[4][4] = hT + w*c;
    }
}

/* =========================================================================
   
   ========================================================================= */
__device__ void ComputeFhat_Pure_dev(const int s, const int k, const int j, const int i,
                                     const int nx, const int ny, const int nz, const Real gamma,
                                     const Real *U_global, const int *did, Real Fhat[]) {
    const int h[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    Real UL[DIMU], UR[DIMU];
    int idxL = IdxNode(k, j, i, ny, nx) * DIMU;
    int idxR = IdxNode(k + h[s][2], j + h[s][1], i + h[s][0], ny, nx) * DIMU;
    for(int n=0; n<DIMU; ++n) { UL[n] = U_global[idxL+n]; UR[n] = U_global[idxR+n]; }

    Real Uo[DIMUo];
    SymmetricAverage_dev(gamma, UL, UR, Uo);

    Real Lambda[DIMU], L[DIMU][DIMU], R_mat[DIMU][DIMU];
    Eigenvalue_dev(s, Uo, Lambda);
    EigenvectorL_dev(s, gamma, Uo, L);
    EigenvectorR_dev(s, Uo, R_mat);

    Real LambdaP[DIMU], LambdaN[DIMU];
    StegerWarming_dev(Lambda, LambdaP, LambdaN);

    Real W[6][DIMU];
    for (int n_idx = WENO_SL, m = 0; n_idx <= WENO_SR; ++n_idx, ++m) {
        int idx = IdxNode(k + n_idx * h[s][2], j + n_idx * h[s][1], i + n_idx * h[s][0], ny, nx) * DIMU;
        for (int r = 0; r < DIMU; ++r) {
            W[m][r] = 0.0;
            for (int c = 0; c < DIMU; ++c) W[m][r] += L[r][c] * U_global[idx+c];
        }
    }

    Real HP[5][DIMU], HN[5][DIMU];
    for (int n_idx = 0, m = 0; m < 5; n_idx += 1, ++m) {
        for (int r = 0; r < DIMU; ++r) HP[m][r] = LambdaP[r] * W[n_idx][r];
    }
    for (int n_idx = 5, m = 0; m < 5; n_idx -= 1, ++m) {
        for (int r = 0; r < DIMU; ++r) HN[m][r] = LambdaN[r] * W[n_idx][r];
    }

    Real HhatP[DIMU], HhatN[DIMU];

    #if WENO_SCHEME == 1
        WENO5_dev(HP, HhatP);
        WENO5_dev(HN, HhatN);
    #else
        WENO3_dev(HP, HhatP);
        WENO3_dev(HN, HhatN);
    #endif

    for (int r = 0; r < DIMU; ++r) {
        Fhat[r] = 0.0;
        for (int c = 0; c < DIMU; ++c) Fhat[r] += R_mat[r][c] * (HhatP[c] + HhatN[c]);
    }
}


__global__ void ComputeFhat_Structured_kernel(
    int nx, int ny, int nz, int s, Real gamma,
    const Real * __restrict__ U,
    Real * __restrict__ Fhat,
    const int * __restrict__ did
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Nnodes = nx * ny * nz;
    if (idx >= Nnodes) return;

    // Prevent Segfaults on empty sweeps!
    if (s == 0 && nx <= 3) return;
    if (s == 1 && ny <= 3) return;
    if (s == 2 && nz <= 3) return;

    int i = idx % nx;
    int j = (idx / nx) % ny;
    int k = idx / (nx * ny);

    
    if (nx > 3 && (i < 2 || i >= nx - 3)) return;
    if (ny > 3 && (j < 2 || j >= ny - 3)) return;
    if (nz > 3 && (k < 2 || k >= nz - 3)) return;

    int out_idx = idx * 5;

    // Determine if this cell is safely inside the interior computational core
    int is_interior = (nx <= 3 || (i >= 3 && i < nx - 3)) && 
                      (ny <= 3 || (j >= 3 && j < ny - 3)) && 
                      (nz <= 3 || (k >= 3 && k < nz - 3));

   
    
   

    Real local_Fhat[5] = {0.0};
    ComputeFhat_Pure_dev(s, k, j, i, nx, ny, nz, gamma, U, did, local_Fhat);

    // // === GPU FLUX DIAGNOSTIC INJECTION START ===
    // // We check face i=2 (FhatL for cell 3) and face i=3 (FhatR for cell 3)
    // if (s == 0 && j == 3 && k == 0) {
    //     if (i == 2) { 
    //         printf("[GPU WENO-5] i=3, j=3 | FhatL: rho=%.6f | rhou=%.6f | rhov=%.6f | E=%.6f\n", 
    //                local_Fhat[0], local_Fhat[1], local_Fhat[2], local_Fhat[4]);
    //     }
    //     if (i == 3) { 
    //         printf("[GPU WENO-5] i=3, j=3 | FhatR: rho=%.6f | rhou=%.6f | rhov=%.6f | E=%.6f\n", 
    //                local_Fhat[0], local_Fhat[1], local_Fhat[2], local_Fhat[4]);
    //     }
    // }
    // // === GPU FLUX DIAGNOSTIC INJECTION END ===

    for(int v = 0; v < 5; ++v) {
        Fhat[out_idx + v] = local_Fhat[v];
    }
}

/* =========================================================================
   LAUNCHER
   ========================================================================= */
extern "C"
int LaunchComputeFhatGPU(int nx, int ny, int nz, int s, Real gamma, const Real* U_h, Real* Fhat_h, const int* did)
{
    int Nnodes = nx * ny * nz;
    int blocks = (Nnodes + TPB - 1) / TPB;

    
    ComputeFhat_Structured_kernel<<<blocks, TPB, 0, computeStream>>>(nx, ny, nz, s, gamma, U_h, Fhat_h, did);

    return 0;
}