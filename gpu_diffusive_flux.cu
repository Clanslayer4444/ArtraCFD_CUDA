#include <cuda_runtime.h>
#include <stdio.h>
#include "commons.h"
#include "gpu_diffusive_flux.h"

extern cudaStream_t computeStream;

#define TPB_X 8
#define TPB_Y 8
#define TPB_Z 4
#define DIMU 5


__device__ __forceinline__ void GetPrimitive_dev(const Real *U, Real gamma, Real gasR, Real *prim) {
    const Real rho = U[0];
    const Real inv_rho = 1.0 / rho;
    const Real u = U[1] * inv_rho;
    const Real v = U[2] * inv_rho;
    const Real w = U[3] * inv_rho;
    const Real E = U[4]; 

    const Real q = 0.5 * rho * (u*u + v*v + w*w);
    const Real p = (gamma - 1.0) * (E - q);
    const Real T = p * inv_rho / gasR;

    prim[0] = rho; prim[1] = u; prim[2] = v;
    prim[3] = w; prim[4] = p; prim[5] = T;
}

/* -------------------------------------------------------------------------
   Device Helper: Sutherland's Law
   ------------------------------------------------------------------------- */
__device__ __forceinline__ Real Viscosity_dev(Real T) {
    return 1.458e-6 * pow(T, 1.5) / (T + 110.4);
}

__device__ __forceinline__ int Idx(int k, int j, int i, int slice, int row) {
    return k * slice + j * row + i;
}

/* -------------------------------------------------------------------------
   Kernel: ComputeFvhat (Viscous Flux)
   ------------------------------------------------------------------------- */
__global__ void ComputeFvhat_kernel(
    int nx, int ny, int nz, int s, 
    Real gamma, Real gasR, Real refMu, Real refT, Real Prandtl,
    Real dx, Real dy, Real dz,
    const Real * __restrict__ U,   // VRAM Persistent State
    Real * __restrict__ Fvhat      // VRAM Persistent Flux
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx - 1 || j >= ny - 1 || k >= nz) return;
    if (i < 1 || j < 1) return;
    
    /* 2D Z-Shield: Do not compute flux in Z direction for 2D grids */
    if (nz == 1 && s == 2) {
        int idx_z = Idx(k, j, i, nx * ny, nx);
        for(int n=0; n<DIMU; ++n) Fvhat[idx_z*DIMU + n] = 0.0;
        return;
    }
    /* Shield Z boundaries for 3D grids */
    if (nz > 1 && (k >= nz - 1 || k < 1)) return;

    int row = nx;
    int slice = nx * ny;
    int idx = Idx(k, j, i, slice, row);

    if (refMu <= 0.0) {
        for(int n=0; n<DIMU; ++n) Fvhat[idx*DIMU + n] = 0.0;
        return;
    }

    int idxL = idx, idxR = 0;
    int idxL_T1a=0, idxL_T1b=0, idxR_T1a=0, idxR_T1b=0;
    int idxL_T2a=0, idxL_T2b=0, idxR_T2a=0, idxR_T2b=0;
    Real dd_norm, dd_t1, dd_t2; 

    int is3D = (nz > 1);

    if (s == 0) {
        idxR = Idx(k, j, i+1, slice, row);
        idxL_T1a = Idx(k, j+1, i, slice, row); idxL_T1b = Idx(k, j-1, i, slice, row);
        idxR_T1a = Idx(k, j+1, i+1, slice, row); idxR_T1b = Idx(k, j-1, i+1, slice, row);
        if (is3D) {
            idxL_T2a = Idx(k+1, j, i, slice, row); idxL_T2b = Idx(k-1, j, i, slice, row);
            idxR_T2a = Idx(k+1, j, i+1, slice, row); idxR_T2b = Idx(k-1, j, i+1, slice, row);
        }
        dd_norm = 1.0/dx; dd_t1 = 1.0/dy; dd_t2 = 1.0/dz;
    }
    else if (s == 1) {
        idxR = Idx(k, j+1, i, slice, row);
        idxL_T1a = Idx(k, j, i+1, slice, row); idxL_T1b = Idx(k, j, i-1, slice, row);
        idxR_T1a = Idx(k, j+1, i+1, slice, row); idxR_T1b = Idx(k, j+1, i-1, slice, row);
        if (is3D) {
            idxL_T2a = Idx(k+1, j, i, slice, row); idxL_T2b = Idx(k-1, j, i, slice, row);
            idxR_T2a = Idx(k+1, j+1, i, slice, row); idxR_T2b = Idx(k-1, j+1, i, slice, row);
        }
        dd_norm = 1.0/dy; dd_t1 = 1.0/dx; dd_t2 = 1.0/dz;
    }
    else {
        idxR = Idx(k+1, j, i, slice, row);
        idxL_T1a = Idx(k, j, i+1, slice, row); idxL_T1b = Idx(k, j, i-1, slice, row);
        idxR_T1a = Idx(k+1, j, i+1, slice, row); idxR_T1b = Idx(k+1, j, i-1, slice, row);
        idxL_T2a = Idx(k, j+1, i, slice, row); idxL_T2b = Idx(k, j-1, i, slice, row);
        idxR_T2a = Idx(k+1, j+1, i, slice, row); idxR_T2b = Idx(k+1, j-1, i, slice, row);
        dd_norm = 1.0/dz; dd_t1 = 1.0/dx; dd_t2 = 1.0/dy;
    }

    #define LOAD_PRIM(index, var_out) { \
        Real U_local[DIMU]; \
        for(int l=0; l<DIMU; ++l) U_local[l] = U[(index)*DIMU + l]; \
        GetPrimitive_dev(U_local, gamma, gasR, var_out); \
    }

    Real P_L[6], P_R[6], P_LT1a[6], P_LT1b[6], P_RT1a[6], P_RT1b[6];
    LOAD_PRIM(idxL, P_L); LOAD_PRIM(idxR, P_R);
    LOAD_PRIM(idxL_T1a, P_LT1a); LOAD_PRIM(idxL_T1b, P_LT1b);
    LOAD_PRIM(idxR_T1a, P_RT1a); LOAD_PRIM(idxR_T1b, P_RT1b);

    Real P_LT2a[6], P_LT2b[6], P_RT2a[6], P_RT2b[6];
    if (is3D) {
        LOAD_PRIM(idxL_T2a, P_LT2a); LOAD_PRIM(idxL_T2b, P_LT2b);
        LOAD_PRIM(idxR_T2a, P_RT2a); LOAD_PRIM(idxR_T2b, P_RT2b);
    }

    #define TRANS_GRAD(idx_v, dd) (0.25 * ((P_RT1a[idx_v] + P_LT1a[idx_v]) - (P_RT1b[idx_v] + P_LT1b[idx_v])) * dd)
    #define TRANS_GRAD_3D(idx_v, dd) (0.25 * ((P_RT2a[idx_v] + P_LT2a[idx_v]) - (P_RT2b[idx_v] + P_LT2b[idx_v])) * dd)

    Real du_dn = (P_R[1] - P_L[1]) * dd_norm;
    Real dv_dn = (P_R[2] - P_L[2]) * dd_norm;
    Real dw_dn = (P_R[3] - P_L[3]) * dd_norm;
    Real dT_dn = (P_R[5] - P_L[5]) * dd_norm;

    Real du_dt1 = TRANS_GRAD(1, dd_t1);
    Real dv_dt1 = TRANS_GRAD(2, dd_t1);
    Real dw_dt1 = TRANS_GRAD(3, dd_t1);

    Real du_dt2 = 0.0, dv_dt2 = 0.0, dw_dt2 = 0.0;
    if (is3D) {
        du_dt2 = TRANS_GRAD_3D(1, dd_t2);
        dv_dt2 = TRANS_GRAD_3D(2, dd_t2);
        dw_dt2 = TRANS_GRAD_3D(3, dd_t2);
    }

    Real uhat = 0.5 * (P_L[1] + P_R[1]);
    Real vhat = 0.5 * (P_L[2] + P_R[2]);
    Real what = 0.5 * (P_L[3] + P_R[3]);
    Real That = 0.5 * (P_L[5] + P_R[5]);

    Real mu = refMu * Viscosity_dev(That * refT);
    Real cv = gasR / (gamma - 1.0);
    Real heatK = gamma * cv * mu / Prandtl;

    Real du_dx = 0.0, dv_dy = 0.0, dw_dz = 0.0;
    if (s == 0)      { du_dx = du_dn;  dv_dy = dv_dt1; dw_dz = dw_dt2; }
    else if (s == 1) { du_dx = du_dt1; dv_dy = du_dn;  dw_dz = dw_dt2; } 
    else             { du_dx = du_dt1; dv_dy = du_dt2; dw_dz = du_dn;  }

    Real divV = du_dx + dv_dy + dw_dz;
    Real f1 = 0.0, f2 = 0.0, f3 = 0.0, f4 = 0.0;

    if (s == 0) {
        f1 = mu * (2.0*du_dn - (2.0/3.0)*divV);
        f2 = mu * (dv_dn + du_dt1);
        f3 = mu * (dw_dn + du_dt2);
    }
    else if (s == 1) {
        f1 = mu * (du_dn + dv_dt1);
        f2 = mu * (2.0*dv_dn - (2.0/3.0)*divV);
        f3 = mu * (dw_dn + dv_dt2);
    }
    else {
        f1 = mu * (du_dn + dw_dt1);
        f2 = mu * (dv_dn + dw_dt2);
        f3 = mu * (2.0*dw_dn - (2.0/3.0)*divV);
    }
    f4 = heatK * dT_dn + f1*uhat + f2*vhat + f3*what;

    Fvhat[idx*DIMU + 0] = 0.0;
    Fvhat[idx*DIMU + 1] = f1;
    Fvhat[idx*DIMU + 2] = f2;
    Fvhat[idx*DIMU + 3] = f3;
    Fvhat[idx*DIMU + 4] = f4;
}

/* ===============
   LAUNCHER...VRAM persistent, ZERO PCIe overhead
   ==================== */
extern "C"
int LaunchComputeFvhatGPU(
    int nx, int ny, int nz, int s,
    Real gamma, Real gasR, Real refMu, Real refT, Real Prandtl,
    Real dx, Real dy, Real dz,
    const Real *d_U,    /* Must be device pointer! */
    Real *d_Fvhat       /* Must be device pointer! */
)
{
    dim3 threads(TPB_X, TPB_Y, TPB_Z);
    if (nz == 1) threads = dim3(16, 16, 1);

    dim3 blocks(
        (nx + threads.x - 1) / threads.x,
        (ny + threads.y - 1) / threads.y,
        (nz + threads.z - 1) / threads.z
    );

    ComputeFvhat_kernel<<<blocks, threads, 0, computeStream>>>(
        nx, ny, nz, s,
        gamma, gasR, refMu, refT, Prandtl,
        dx, dy, dz,
        d_U, d_Fvhat
    );
    
    return 0;
}