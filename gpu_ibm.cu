#include <cuda_runtime.h>
#include <stdio.h>
#include "commons.h"
#include "gpu_ibm.cuh"
extern cudaStream_t computeStream;

/* Global device pointer for the stencil map */
IBM_Stencil *d_ibm_map = NULL;
int d_num_stencils = 0;




__global__ void AuditDeviceMap_kernel(IBM_Stencil *d_map, int num_stencils) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double pos_ny = 0.0;
        double neg_ny = 0.0;
        for(int i = 0; i < num_stencils; ++i) {
            if (d_map[i].N[1] > 0.0) pos_ny += d_map[i].N[1];
            if (d_map[i].N[1] < 0.0) neg_ny += d_map[i].N[1];
        }
        printf("[GPU DIAGNOSTIC] sizeof(IBM_Stencil) = %d bytes\n", (int)sizeof(IBM_Stencil));
        printf("[GPU DIAGNOSTIC] Device Pos Ny: %f | Neg Ny: %f\n\n", pos_ny, neg_ny);
    }
}


/* --- Math Helpers --- */
__device__ void MapPrimitive_GPU(const Real gamma, const Real gasR, const Real U[], Real Uo[]) {
    // Safety check for vacuum or uninitialized nodes
    if (U[0] < 1.0e-6) {
        Uo[0] = 1.0; // Default reference density
        Uo[1] = 0.0; Uo[2] = 0.0; Uo[3] = 0.0; // Zero velocity
        Uo[4] = 1.0; // Default reference pressure
        Uo[5] = 1.0; // Default reference temperature
        return;
    }

    Uo[0] = U[0];
    Uo[1] = U[1] / U[0];
    Uo[2] = U[2] / U[0];
    Uo[3] = U[3] / U[0];
    Uo[4] = (U[4] - 0.5 * (U[1]*U[1] + U[2]*U[2] + U[3]*U[3]) / U[0]) * (gamma - 1.0);
    Uo[5] = Uo[4] / (Uo[0] * gasR);
}

// Inside ApplyIBM_GPU_kernel (Step 3: Write Back)
__device__ void MapConservative_GPU(const Real gamma, const Real gasR, const Real Uo[], Real U[]) {
    // Safety: Ensure Temperature is not zero
    Real T = (Uo[5] < 1.0e-6) ? 1.0 : Uo[5]; 
    Real rho = Uo[4] / (T * gasR);

    // Apply clamp
    if (rho < 1.0e-6) rho = 1.0e-6;

    U[0] = rho;
    U[1] = U[0] * Uo[1];
    U[2] = U[0] * Uo[2];
    U[3] = U[0] * Uo[3];
    U[4] = 0.5 * U[0] * (Uo[1]*Uo[1] + Uo[2]*Uo[2] + Uo[3]*Uo[3]) + Uo[4] / (gamma - 1.0);
}

__device__ void Cross_GPU(const Real a[3], const Real b[3], Real c[3]) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ Real Dot_GPU(const Real a[3], const Real b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

__device__ void OrthogonalSpace_GPU(const Real n[3], Real ta[3], Real tb[3]) {
    Real threshold = 0.1;
    if (fabs(n[0]) > threshold || fabs(n[1]) > threshold) {
        ta[0] = -n[1]; ta[1] = n[0]; ta[2] = 0.0;
    } else {
        ta[0] = 0.0; ta[1] = -n[2]; ta[2] = n[1];
    }
    Real magA = sqrt(ta[0]*ta[0] + ta[1]*ta[1] + ta[2]*ta[2]);
    ta[0] /= magA; ta[1] /= magA; ta[2] /= magA;
    Cross_GPU(n, ta, tb);
}


__global__ void ApplyIBM_GPU_kernel(
    IBM_Stencil *map, int num_stencils,
    Real *U, Real gamma, Real gasR,
    int target_mirror // 1 for Boundary, 0 for Padding
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_stencils) return;

    IBM_Stencil s = map[tid];
    
    // RACE CONDITION FIX: Only process the requested layer
    if (s.is_mirror != target_mirror) return;

    int solid_idx = s.ghost_idx;

    // Pure Primitive Variable IDW (Matches CPU exactly).....
    Real Uo_sum[6] = {0.0};
    Real valid_weight_sum = 0.0;
    
    for (int n = 0; n < s.num_neighbors; n++) {
        int fluid_idx = s.neighbor_indices[n];
        Real weight = s.weights[n];
        
        Real U_local[5];
        for(int v=0; v<5; ++v) U_local[v] = U[fluid_idx * 5 + v];
        
        // Skipping empty memory (safety net)
        if (U_local[0] < 1.0e-6) continue; 
        
        Real Uo_fluid[6];
        MapPrimitive_GPU(gamma, gasR, U_local, Uo_fluid);
        for(int v=0; v<6; ++v) Uo_sum[v] += Uo_fluid[v] * weight;
        valid_weight_sum += weight;
    }
    
    if (valid_weight_sum > 1.0e-12) {
        for(int v=0; v<6; ++v) Uo_sum[v] /= valid_weight_sum;
    } else {
        return; 
    }

    if (s.is_mirror) {
        // CPU ReconstructFlow Logic
        Real Uo_O[6];
        Real Vs[3] = {0.0, 0.0, 0.0};
        
        Real rx = s.pO[0] - s.poly_O[0];
        Real ry = s.pO[1] - s.poly_O[1];
        Real rz = s.pO[2] - s.poly_O[2];
        
        Vs[0] = s.poly_V[0] + (s.poly_W[1]*rz - s.poly_W[2]*ry);
        Vs[1] = s.poly_V[1] + (s.poly_W[2]*rx - s.poly_W[0]*rz);
        Vs[2] = s.poly_V[2] + (s.poly_W[0]*ry - s.poly_W[1]*rx);

        if (s.poly_cf > 0.0) {
            Uo_O[1] = Vs[0];
            Uo_O[2] = Vs[1];
            Uo_O[3] = Vs[2];
        } else {
            Real V[3] = {Uo_sum[1], Uo_sum[2], Uo_sum[3]};
            Real N[3] = {s.N[0], s.N[1], s.N[2]};
            Real v_dot_n = V[0]*N[0] + V[1]*N[1] + V[2]*N[2];
            Real vs_dot_n = Vs[0]*N[0] + Vs[1]*N[1] + Vs[2]*N[2];
            
            Uo_O[1] = N[0]*vs_dot_n + (V[0] - v_dot_n*N[0]);
            Uo_O[2] = N[1]*vs_dot_n + (V[1] - v_dot_n*N[1]);
            Uo_O[3] = N[2]*vs_dot_n + (V[2] - v_dot_n*N[2]);
        }

        Uo_O[4] = Uo_sum[4]; 
        Uo_O[5] = (s.poly_T <= 0.0) ? Uo_sum[5] : s.poly_T; 

        // CPU DoMethodOfImage Logic
        Real Uo_G[6];
        Uo_G[1] = 2.0 * Uo_O[1] - Uo_sum[1];
        Uo_G[2] = 2.0 * Uo_O[2] - Uo_sum[2];
        Uo_G[3] = 2.0 * Uo_O[3] - Uo_sum[3];
        Uo_G[4] = Uo_sum[4];
        Uo_G[5] = Uo_sum[5];
        
        if (Uo_G[5] > 0.0) { 
            Uo_G[0] = Uo_G[4] / (Uo_G[5] * gasR);
            MapConservative_GPU(gamma, gasR, Uo_G, &U[solid_idx * 5]);
        }
        
    } else {
        // CPU Deep Padding Logic
        if (Uo_sum[5] > 0.0) {
            Uo_sum[0] = Uo_sum[4] / (Uo_sum[5] * gasR);
            MapConservative_GPU(gamma, gasR, Uo_sum, &U[solid_idx * 5]);
        }
    }
}

/* --- Host Interface --- */
extern "C" void InitIBM_GPU(IBM_Stencil *h_map, int num_stencils) {
    if (num_stencils <= 0 || h_map == NULL) {
        printf("DEBUG: IBM Init skipped (Stencils: %d, Pointer: %p)\n", num_stencils, (void*)h_map);
        return;
    }
    
    // --- GEOMETRY SYMMETRY AUDIT ---
    // using strict double precision here just to catch floating-point noise in the map..
    double pos_ny_sum = 0.0;
    double neg_ny_sum = 0.0;
    double pos_weight_sum = 0.0;
    double neg_weight_sum = 0.0;

    for (int i = 0; i < num_stencils; i++) {
        if (h_map[i].N[1] > 0.0) {
            pos_ny_sum += (double)h_map[i].N[1];
            pos_weight_sum += (double)h_map[i].weight_sum;
        }
        if (h_map[i].N[1] < 0.0) {
            neg_ny_sum += (double)h_map[i].N[1];
            neg_weight_sum += (double)h_map[i].weight_sum;
        }
    }

    printf("\n=== IBM GEOMETRY SYMMETRY AUDIT ===\n");
    printf("Total Stencils: %d\n", num_stencils);
    printf("Positive Y-Normals Sum: %15.9f\n", pos_ny_sum);
    printf("Negative Y-Normals Sum: %15.9f\n", neg_ny_sum);
    printf("Normals Abs Difference: %15.9f\n", fabs(pos_ny_sum) - fabs(neg_ny_sum));
    printf("Weights Abs Difference: %15.9f\n", fabs(pos_weight_sum) - fabs(neg_weight_sum));
    printf("===================================\n\n");

    d_num_stencils = num_stencils;
    if (d_ibm_map) cudaFree(d_ibm_map);
    
    cudaError_t err = cudaMalloc(&d_ibm_map, num_stencils * sizeof(IBM_Stencil));
    if (err != cudaSuccess) { printf("CRITICAL: IBM Malloc failed: %s\n", cudaGetErrorString(err)); exit(1); }
    
    err = cudaMemcpy(d_ibm_map, h_map, num_stencils * sizeof(IBM_Stencil), cudaMemcpyHostToDevice);

    AuditDeviceMap_kernel<<<1, 1>>>(d_ibm_map, num_stencils);
    cudaDeviceSynchronize();

    if (err != cudaSuccess) { printf("CRITICAL: IBM Memcpy failed: %s\n", cudaGetErrorString(err)); exit(1); }
    
    printf(">> GPU IBM Map Loaded: %d stencils\n", num_stencils);
}

extern "C" void ApplyIBM_GPU(Real *d_U, Real gamma, Real gasR) {
    if (d_num_stencils == 0 || d_ibm_map == NULL) return;

    
    int threads = 256;
    int blocks = (d_num_stencils + threads - 1) / threads;

    // Pass 1: Solve the Boundary Layer (is_mirror = 1)
    ApplyIBM_GPU_kernel<<<blocks, threads, 0, computeStream>>>(d_ibm_map, d_num_stencils, d_U, gamma, gasR, 1);

    // Pass 2: Solve the Deep Padding Layer (is_mirror = 0). Pass 2's IDW stencils
    // may read ghost cells that Pass 1 just wrote, so Pass 2 must not begin
    // reading until Pass 1 has finished writing.
    //
    // NOTE (profiling finding, July 2026): this ordering requirement is
    // already guaranteed WITHOUT any explicit sync, because neither kernel launch
    // specifies a stream, so both go to the default stream (stream 0). CUDA's
    // same-stream ordering guarantee ensures Pass 2 cannot start until Pass 1
    // completes, and that Pass 2 will see all of Pass 1's global-memory writes.
    // The cudaDeviceSynchronize() calls previously here provided no additional
    // correctness -- they only blocked the host CPU on every single call and
    // were responsible for ~93% of total wall-clock runtime (confirmed via
    // `nsys stats --report cudaapisum`: cudaDeviceSynchronize = 176s / 188s
    // total, 28273 calls, avg 6.2ms/call). Removed for that reason.
    // If pass 1/2 are ever moved to different (non-default) streams, an
    // explicit cudaStreamSynchronize() (not cudaDeviceSynchronize()) or a
    // CUDA event dependency between those specific streams must be
    // reintroduced here, or this race condition will return.
    ApplyIBM_GPU_kernel<<<blocks, threads, 0, computeStream>>>(d_ibm_map, d_num_stencils, d_U, gamma, gasR, 0);
}      

extern "C" void FreeIBM_GPU() {
    if (d_ibm_map) {
        cudaFree(d_ibm_map);
        d_ibm_map = NULL;
    }
    d_num_stencils = 0;
}