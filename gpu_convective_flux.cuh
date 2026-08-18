#ifndef GPU_CONVECTIVE_FLUX_CUH
#define GPU_CONVECTIVE_FLUX_CUH

#include "commons.h"

#ifdef __cplusplus
extern "C" {
#endif

int LaunchComputeFhatGPU(
    int Nflux,
    int s,
    int sScheme,
    int fluxSplit,
    Real gamma,
    const int* faceL_h,
    const int* faceR_h,
    const Real* U_h,
    Real* Fhat_h
);

#ifdef __cplusplus
}
#endif

#endif
