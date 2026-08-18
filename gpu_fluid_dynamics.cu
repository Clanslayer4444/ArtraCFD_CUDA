#include <cuda_runtime.h>
#include <stdio.h>
#include "commons.h"
#include "gpu_fluid_dynamics.h"
#include "gpu_convective_flux.h"
#include "gpu_diffusive_flux.h"
#include <stdlib.h>

__constant__ int c_ng[3];
__constant__ int c_typeBC[6];

extern "C" void SyncBoundaryParams(int ng0, int ng1, int ng2, int bc0, int bc1, int bc2, int bc3, int bc4, int bc5) {
    int h_ng[3] = {ng0, ng1, ng2};
    int h_bc[6] = {bc0, bc1, bc2, bc3, bc4, bc5};
    cudaMemcpyToSymbol(c_ng, h_ng, 3 * sizeof(int));
    cudaMemcpyToSymbol(c_typeBC, h_bc, 6 * sizeof(int));
}


Real *d_U = NULL;
Real *d_U0 = NULL;
Real *d_Fhat = NULL;
Real *d_Fvhat = NULL;
Real *d_varBC = NULL;
int  *d_did = NULL;
cudaStream_t computeStream;
cudaStream_t ioStream;
Real *h_pinned_U = NULL;

extern "C" void LaunchBoundaryKernel(int nx, int ny, int nz, const Partition *part, const Model *model);
extern "C" void ApplyIBM_GPU(Real *d_U, Real gamma, Real gasR);
extern "C" Real* GPU_Get_U_TN() { return d_U; }



__device__ void MapPrimitive_dev(const Real gamma, const Real gasR, const Real U[], Real Uo[]) {
    Uo[0] = U[0];
    Uo[1] = U[1] / U[0];
    Uo[2] = U[2] / U[0];
    Uo[3] = U[3] / U[0];
    Uo[4] = (U[4] - 0.5 * (U[1]*U[1] + U[2]*U[2] + U[3]*U[3]) / U[0]) * (gamma - 1.0);
    Uo[5] = Uo[4] / (Uo[0] * gasR);
}

__device__ void MapConservative_dev(const Real gamma, const Real Uo[], Real U[]) {
    U[0] = Uo[0];
    U[1] = Uo[0] * Uo[1];
    U[2] = Uo[0] * Uo[2];
    U[3] = Uo[0] * Uo[3];
    U[4] = 0.5 * Uo[0] * (Uo[1]*Uo[1] + Uo[2]*Uo[2] + Uo[3]*Uo[3]) + Uo[4] / (gamma - 1.0);
}

#define CFL_TPB 256

__global__ void ComputeCFL_GPU_kernel(
    int nx, int ny, int nz,
    int xmin, int xmax, int ymin, int ymax, int zmin, int zmax,
    Real gamma, Real gasR,
    const Real *U, const int *did,
    Real *blockVmax)
{
    __shared__ Real sVx[CFL_TPB];
    __shared__ Real sVy[CFL_TPB];
    __shared__ Real sVz[CFL_TPB];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int total = nx * ny * nz;

    Real Vx = 0.0, Vy = 0.0, Vz = 0.0;

    if (gid < total) {
        int i = gid % nx;
        int j = (gid / nx) % ny;
        int k = gid / (nx * ny);

        if (i >= xmin && i < xmax && j >= ymin && j < ymax && k >= zmin && k < zmax) {
            if (did[gid] == 0) {
                Real Uo[6];
                MapPrimitive_dev(gamma, gasR, &U[gid*5], Uo);
                Real c = sqrt(gamma * gasR * Uo[5]);
                Vx = fabs(Uo[1]) + c;
                Vy = fabs(Uo[2]) + c;
                Vz = fabs(Uo[3]) + c;
            }
        }
    }

    sVx[tid] = Vx; sVy[tid] = Vy; sVz[tid] = Vz;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sVx[tid + stride] > sVx[tid]) sVx[tid] = sVx[tid + stride];
            if (sVy[tid + stride] > sVy[tid]) sVy[tid] = sVy[tid + stride];
            if (sVz[tid + stride] > sVz[tid]) sVz[tid] = sVz[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockVmax[blockIdx.x*3 + 0] = sVx[0];
        blockVmax[blockIdx.x*3 + 1] = sVy[0];
        blockVmax[blockIdx.x*3 + 2] = sVz[0];
    }
}

extern "C" void LaunchComputeCFL_GPU(
    int nx, int ny, int nz,
    int xmin, int xmax, int ymin, int ymax, int zmin, int zmax,
    Real gamma, Real gasR,
    Real *Vmax_out)
{
    int total = nx * ny * nz;
    int blocks = (total + CFL_TPB - 1) / CFL_TPB;

    static Real *d_blockVmax = NULL;
    static int allocated_blocks = 0;
    if (blocks > allocated_blocks) {
        if (d_blockVmax) cudaFree(d_blockVmax);
        cudaMalloc(&d_blockVmax, blocks * 3 * sizeof(Real));
        allocated_blocks = blocks;
    }

    ComputeCFL_GPU_kernel<<<blocks, CFL_TPB, 0, computeStream>>>(
        nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax,
        gamma, gasR, d_U, d_did, d_blockVmax);

    Real *h_blockVmax = (Real*)malloc(blocks * 3 * sizeof(Real));
    cudaMemcpy(h_blockVmax, d_blockVmax, blocks * 3 * sizeof(Real), cudaMemcpyDeviceToHost);

    Vmax_out[0] = 0.0; Vmax_out[1] = 0.0; Vmax_out[2] = 0.0;
    for (int b = 0; b < blocks; ++b) {
        if (h_blockVmax[b*3+0] > Vmax_out[0]) Vmax_out[0] = h_blockVmax[b*3+0];
        if (h_blockVmax[b*3+1] > Vmax_out[1]) Vmax_out[1] = h_blockVmax[b*3+1];
        if (h_blockVmax[b*3+2] > Vmax_out[2]) Vmax_out[2] = h_blockVmax[b*3+2];
    }
    free(h_blockVmax);
}

__global__ void GPU_InitField_kernel(
    int nx, int ny, int nz, 
    Real *U, 
    GPU_InitParams params // New parameter struct
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Nnodes = nx * ny * nz;
    if (idx >= Nnodes) return;

    // Direct calculation based on parameters, 
    Real rho = params.rho;
    Real u = params.u;
    Real p = params.p;
    Real gamma = 1.4;

    U[idx * 5 + 0] = rho;
    U[idx * 5 + 1] = rho * u;
    U[idx * 5 + 2] = 0.0;
    U[idx * 5 + 3] = 0.0;
    U[idx * 5 + 4] = p / (gamma - 1.0) + 0.5 * rho * (u * u);
}

extern "C" void LaunchGPUInitialization(int nx, int ny, int nz, Real *d_U, GPU_InitParams params) {
    int Nnodes = nx * ny * nz;
    int TPB = 256;
    int blocks = (Nnodes + TPB - 1) / TPB;
    
    GPU_InitField_kernel<<<blocks, TPB>>>(nx, ny, nz, d_U, params);
    cudaDeviceSynchronize();
}

// Direct VRAM array, bypassing the misaligned CPU struct....
__device__ int gpu_b_types[6];

extern "C" void SetGPUBoundaryTypes(int w, int e, int s, int n, int b, int t) {
    int host_b_types[6] = {w, e, s, n, b, t};
    cudaMemcpyToSymbol(gpu_b_types, host_b_types, 6 * sizeof(int));
}



__global__ void TreatBoundary_GPU_kernel(
    int Nnodes, int nx, int ny, int nz, 
    int x_min, int x_max, int y_min, int y_max, int z_min, int z_max,
    int pwb_type, int peb_type, int psb_type, int pnb_type, int pfb_type, int pbb_type,
    Real gamma, Real gasR, const Real *varBC, Real *U
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nnodes) return;

    int i = idx % nx;
    int j = (idx / nx) % ny;
    int k = idx / (nx * ny);

    int is_ghost = 0;
  
    if (nx > 3 && (i < x_min || i >= x_max)) is_ghost = 1;
    if (ny > 3 && (j < y_min || j >= y_max)) is_ghost = 1;
    if (nz > 3 && (k < z_min || k >= z_max)) is_ghost = 1;

    if (!is_ghost) return;

    int p = -1;
    int n_x = 0, n_y = 0, n_z = 0;
    int b_type = -1;

    if (nx > 3 && i < x_min) { p = PWB; n_x = -1; b_type = pwb_type; }
    if (nx > 3 && i >= x_max) { p = PEB; n_x = 1; b_type = peb_type; }
    if (ny > 3 && j < y_min) { p = PSB; n_y = -1; b_type = psb_type; }
    if (ny > 3 && j >= y_max) { p = PNB; n_y = 1; b_type = pnb_type; }
    if (nz > 3 && k < z_min) { p = PFB; n_z = -1; b_type = pfb_type; }
    if (nz > 3 && k >= z_max) { p = PBB; n_z = 1; b_type = pbb_type; }

    if (p == -1) return;

    Real UoGiven[6] = {0};
    for(int v=0; v<6; ++v) UoGiven[v] = varBC[p * 6 + v];

    
    int interior_i = i;
    if (nx > 3) {
        if (i < x_min) interior_i = 2 * x_min - 1 - i;
        else if (i >= x_max) interior_i = 2 * x_max - 1 - i;
    }
    
    int interior_j = j;
    if (ny > 3) {
        if (j < y_min) interior_j = 2 * y_min - 1 - j;
        else if (j >= y_max) interior_j = 2 * y_max - 1 - j;
    }
    
    int interior_k = k;
    if (nz > 3) {
        if (k < z_min) interior_k = 2 * z_min - 1 - k;
        else if (k >= z_max) interior_k = 2 * z_max - 1 - k;
    }
    
    int interior_idx = (interior_k * ny + interior_j) * nx + interior_i;

    // For OUTFLOW (zero-gradient), use cascading copy that exactly reproduces
    // the CPU behavior. The CPU treats boundary condition in layers (r=0,1,2,...)
    // where each ghost layer at distance r copies from its immediate interior
    // neighbor at distance r-1. This is equivalent to: ghosts layers converge to
    // the boundary-adjacent interior value through repeated nearest-neighbor copy.
    //
    // We achieve the same result in a single pass by computing the cascaded index:
    // for a ghost at offset d from the interior boundary, the zero-gradient result
    // is the interior boundary node itself (after d layers of nearest-neighbor copy).
    // This matches CPU output exactly.
    int cascade_i = i;
    if (nx > 3) {
        if (i < x_min) cascade_i = x_min;         /* ghost west of boundary */
        else if (i >= x_max) cascade_i = x_max - 1; /* ghost east of boundary */
    }
    int cascade_j = j;
    if (ny > 3) {
        if (j < y_min) cascade_j = y_min;
        else if (j >= y_max) cascade_j = y_max - 1;
    }
    int cascade_k = k;
    if (nz > 3) {
        if (k < z_min) cascade_k = z_min;
        else if (k >= z_max) cascade_k = z_max - 1;
    }
    int cascade_idx = (cascade_k * ny + cascade_j) * nx + cascade_i;

    if (b_type == INFLOW) { 
        MapConservative_dev(gamma, UoGiven, &U[idx*5]); 
    } 
    else if (b_type == OUTFLOW) { 
        for(int v=0; v<5; ++v) U[idx*5+v] = U[cascade_idx*5+v]; 
    }


    else if (b_type == SLIPWALL) { 
        U[idx*5+0] = U[interior_idx*5+0]; 
        U[idx*5+1] = U[interior_idx*5+1] * (n_x != 0 ? -1.0 : 1.0); 
        U[idx*5+2] = U[interior_idx*5+2] * (n_y != 0 ? -1.0 : 1.0); 
        U[idx*5+3] = U[interior_idx*5+3] * (n_z != 0 ? -1.0 : 1.0); 
        U[idx*5+4] = U[interior_idx*5+4]; 
    }
    else if (b_type == NOSLIPWALL) {
        U[idx*5+0] = U[interior_idx*5+0]; 
        U[idx*5+1] = -U[interior_idx*5+1]; 
        U[idx*5+2] = -U[interior_idx*5+2]; 
        U[idx*5+3] = -U[interior_idx*5+3]; 
        U[idx*5+4] = U[interior_idx*5+4]; 
    }
}





__global__ void UpdateRK3_kernel(
    int Nnodes, int s, int nx, int ny, int nz, Real dt_ds, Real alpha, Real beta,
    const Real * __restrict__ U0,
    const Real * __restrict__ Fhat,
    const Real * __restrict__ Fvhat,
    Real * __restrict__ U_curr,
    const int * __restrict__ did  
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nnodes) return;
//---------------------
    
 

    int i = idx % nx;
    int j = (idx / nx) % ny;
    int k = idx / (nx * ny);

    int stride = 1;
    if (s == 0) stride = 1;
    else if (s == 1) stride = nx;
    else if (s == 2) stride = nx * ny;

    bool is_inner = true;

    
    if (nx > 3 && (i < 3 || i >= nx - 3)) is_inner = false;
    if (ny > 3 && (j < 3 || j >= ny - 3)) is_inner = false;
    if (nz > 3 && (k < 3 || k >= nz - 3)) is_inner = false;

    // Skip flux application entirely if sweeping a collapsed dimension...
    if (s == 0 && nx <= 3) is_inner = false;
    if (s == 1 && ny <= 3) is_inner = false;
    if (s == 2 && nz <= 3) is_inner = false;

    if (did != NULL && did[idx] != 0) is_inner = false;

    for (int v = 0; v < 5; ++v) {
        int mem_idx = idx * 5 + v;
        int back_idx = (idx - stride) * 5 + v;
        Real flux_diff = 0.0;

        if (is_inner) {
            Real flux_fwd = Fhat[mem_idx] - Fvhat[mem_idx];
            Real flux_bwd = Fhat[back_idx] - Fvhat[back_idx];
            flux_diff = dt_ds * (flux_fwd - flux_bwd);
        }

        U_curr[mem_idx] = alpha * U0[mem_idx] + beta * (U_curr[mem_idx] - flux_diff);
    }
}

extern "C" void LaunchBoundaryKernel(int nx, int ny, int nz, const Partition *part, const Model *model) {
    int Nnodes = nx * ny * nz;
    int TPB = 256;
    int blocks = (Nnodes + TPB - 1) / TPB;

    // Dynamically calculate absolute limits based on the CPU solver's internal bounds (PIN)
    int x_min = part->ns[PIN][X][MIN] - part->ns[PAL][X][MIN];
    int x_max = part->ns[PIN][X][MAX] - part->ns[PAL][X][MIN];
    int y_min = part->ns[PIN][Y][MIN] - part->ns[PAL][Y][MIN];
    int y_max = part->ns[PIN][Y][MAX] - part->ns[PAL][Y][MIN];
    int z_min = part->ns[PIN][Z][MIN] - part->ns[PAL][Z][MIN];
    int z_max = part->ns[PIN][Z][MAX] - part->ns[PAL][Z][MIN];

    TreatBoundary_GPU_kernel<<<blocks, TPB, 0, computeStream>>>(
        Nnodes, nx, ny, nz, 
        x_min, x_max, y_min, y_max, z_min, z_max,
        part->typeBC[PWB], part->typeBC[PEB],
        part->typeBC[PSB], part->typeBC[PNB],
        part->typeBC[PFB], part->typeBC[PBB],
        model->gamma, model->gasR, d_varBC, d_U
    );
}

void LaunchTrueRK3GPU_Stage(
    int stage, int s, int nx, int ny, int nz,
    Real dt, Real ds, Real gamma, Real gasR, Real refMu, Real refT, Real Prandtl,
    const Partition *part, const Model *model
) {
    int Nnodes = nx * ny * nz;

    if (stage == 1) cudaMemcpy(d_U0, d_U, Nnodes * 5 * sizeof(Real), cudaMemcpyDeviceToDevice);

    cudaMemset(d_Fhat, 0, Nnodes * 5 * sizeof(Real));
    cudaMemset(d_Fvhat, 0, Nnodes * 5 * sizeof(Real));

    LaunchComputeFhatGPU(nx, ny, nz, s, gamma, d_U, d_Fhat, d_did);
    LaunchComputeFvhatGPU(nx, ny, nz, s, gamma, gasR, refMu, refT, Prandtl, part->d[0], part->d[1], part->d[2], d_U, d_Fvhat);

    int TPB = 256;
    int blocks = (Nnodes + TPB - 1) / TPB;
    Real dt_ds = dt * ds;
    Real alpha = (stage == 1) ? 0.0 : ((stage == 2) ? 0.75 : 1.0/3.0);
    Real beta  = (stage == 1) ? 1.0 : ((stage == 2) ? 0.25 : 2.0/3.0);

    
    UpdateRK3_kernel<<<blocks, TPB, 0, computeStream>>>(Nnodes, s, nx, ny, nz, dt_ds, alpha, beta, d_U0, d_Fhat, d_Fvhat, d_U, d_did);
}

extern "C" void LaunchRK3FullStep(
    int s, int nx, int ny, int nz,
    Real dt, Real ds, Real gamma, Real gasR, Real refMu, Real refT, Real Prandtl,
    const Partition *part, const Model *model
) {

    
    // Stage 1
    LaunchTrueRK3GPU_Stage(1, s, nx, ny, nz, dt, ds, gamma, gasR, refMu, refT, Prandtl, part, model);
    ApplyIBM_GPU(d_U, gamma, gasR);                // 1. Inject high-pressure solid ghost cells
    LaunchBoundaryKernel(nx, ny, nz, part, model); // 2. Enforce outer domain (inlet/outlet) walls

    // Stage 2
    LaunchTrueRK3GPU_Stage(2, s, nx, ny, nz, dt, ds, gamma, gasR, refMu, refT, Prandtl, part, model);
    ApplyIBM_GPU(d_U, gamma, gasR);
    LaunchBoundaryKernel(nx, ny, nz, part, model);

    // Stage 3
    LaunchTrueRK3GPU_Stage(3, s, nx, ny, nz, dt, ds, gamma, gasR, refMu, refT, Prandtl, part, model);
    ApplyIBM_GPU(d_U, gamma, gasR);
    LaunchBoundaryKernel(nx, ny, nz, part, model);
}

extern "C" int GPUFluid_Init(int Nnodes) {
    size_t size = Nnodes * 5 * sizeof(Real);
    if (d_U) cudaFree(d_U); if (d_U0) cudaFree(d_U0);
    if (d_Fhat) cudaFree(d_Fhat); if (d_Fvhat) cudaFree(d_Fvhat);
    if (d_did) cudaFree(d_did); if (d_varBC) cudaFree(d_varBC);

    cudaMalloc(&d_U, size); cudaMalloc(&d_U0, size);
    cudaMalloc(&d_Fhat, size); cudaMalloc(&d_Fvhat, size);
    cudaMalloc(&d_did, Nnodes * sizeof(int));
    cudaMalloc(&d_varBC, 6 * 6 * sizeof(Real));

    cudaMemset(d_U, 0, size); cudaMemset(d_U0, 0, size);
    cudaMemset(d_Fhat, 0, size); cudaMemset(d_Fvhat, 0, size);
    cudaMemset(d_did, 0, Nnodes * sizeof(int));
    cudaMemset(d_varBC, 0, 6 * 6 * sizeof(Real));

    /* Phase 1: dedicated named streams, decoupled from the legacy default
       stream's implicit whole-context sync behavior. */
    cudaStreamCreate(&computeStream);
    cudaStreamCreate(&ioStream);

    /* Persistent pinned (page-locked) host buffer, allocated once instead of
       malloc/free on every I/O sync. Pinning is required for cudaMemcpyAsync
       to actually run asynchronously -- pageable memory silently falls back
       to synchronous behavior regardless of the Async call. */
    if (h_pinned_U) cudaFreeHost(h_pinned_U);
    cudaHostAlloc(&h_pinned_U, size, cudaHostAllocDefault);

    return 0;
}

extern "C" void GPUFluid_Finalize(void) {
    if (d_U == NULL) return;
    if (d_U) cudaFree(d_U); if (d_U0) cudaFree(d_U0);
    if (d_Fhat) cudaFree(d_Fhat); if (d_Fvhat) cudaFree(d_Fvhat);
    if (d_did) cudaFree(d_did); if (d_varBC) cudaFree(d_varBC);
    d_U = d_U0 = d_Fhat = d_Fvhat = NULL;
    d_did = NULL; d_varBC = NULL;

    cudaStreamDestroy(computeStream);
    cudaStreamDestroy(ioStream);
    if (h_pinned_U) { cudaFreeHost(h_pinned_U); h_pinned_U = NULL; }

    printf(">> GPU Memory Released.\n");
}

extern "C" void FetchGPUFluidData(Real *h_U, int Nnodes) {
    size_t size = Nnodes * 5 * sizeof(Real);
    cudaMemcpyAsync(h_pinned_U, d_U, size, cudaMemcpyDeviceToHost, ioStream);
    cudaStreamSynchronize(ioStream);
    memcpy(h_U, h_pinned_U, size);
}

extern "C" void SendGPUFluidData(const Real *h_U, int Nnodes) {
    cudaMemcpy(d_U, h_U, Nnodes * 5 * sizeof(Real), cudaMemcpyHostToDevice);
}

extern "C" void SendGPUBoundaryData(const Real *h_varBC) {
    cudaMemcpy(d_varBC, h_varBC, 6 * 6 * sizeof(Real), cudaMemcpyHostToDevice);
}

extern "C" void SendGPUDomainData(const int *h_did, int Nnodes) {
    cudaMemcpy(d_did, h_did, Nnodes * sizeof(int), cudaMemcpyHostToDevice);
}
