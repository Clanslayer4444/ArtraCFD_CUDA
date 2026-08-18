#ifndef GPU_DIFFUSIVE_FLUX_H
#define GPU_DIFFUSIVE_FLUX_H

#include "cfd_commons.h" /* For the Real type */

#ifdef __cplusplus
extern "C" {
#endif

int LaunchComputeFvhatGPU(
    int nx, int ny, int nz, int s,
    Real gamma, Real gasR, Real refMu, Real refT, Real Prandtl,
    Real dx, Real dy, Real dz,
    const Real *d_U, 
    Real *d_Fvhat
);

#ifdef __cplusplus
}
#endif

#endif