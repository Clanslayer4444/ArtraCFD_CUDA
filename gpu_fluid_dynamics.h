#ifndef ARTRACFD_GPU_FLUID_DYNAMICS_H_
#define ARTRACFD_GPU_FLUID_DYNAMICS_H_

#include "commons.h"


typedef struct {
    Real rho, u, v, w, p;
} GPU_InitParams;

#ifdef __cplusplus
extern "C" {
#endif

void LaunchComputeCFL_GPU(
    int nx, int ny, int nz,
    int xmin, int xmax, int ymin, int ymax, int zmin, int zmax,
    Real gamma, Real gasR,
    Real *Vmax_out);

/* Initialize/Finalize */
int GPUFluid_Init(int Nnodes);
void GPUFluid_Finalize(void);

/* Data Transfer Bridges */
void SendGPUFluidData(const Real *h_U, int Nnodes);
void SendGPUBoundaryData(const Real *h_varBC);
void SendGPUDomainData(const int *h_did, int Nnodes);
void FetchGPUFluidData(Real *h_U, int Nnodes);

/* NEW PHASE 1: Native GPU Initialization Launcher */
void LaunchGPUInitialization(int nx, int ny, int nz, Real *d_U, GPU_InitParams params);

/* Master Launchers */
void LaunchRK3FullStep(
    int s, int nx, int ny, int nz,
    Real dt, Real ds, Real gamma, Real gasR, 
    Real refMu, Real refT, Real Prandtl, 
    const Partition *part, const Model *model
);

#ifdef __cplusplus
}
#endif

#endif