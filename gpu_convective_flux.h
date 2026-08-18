#ifndef ARTRACFD_GPU_CONVECTIVE_FLUX_H_
#define ARTRACFD_GPU_CONVECTIVE_FLUX_H_

#include "commons.h"

#ifdef __cplusplus
extern "C" {
#endif

int LaunchComputeFhatGPU(int nx, int ny, int nz, int s, Real gamma, const Real* U_h, Real* Fhat_h, const int* did);


/* Test Launchers (Stubs for compatibility) */
int LaunchCharacteristicFluxTestGPU(
    int s,
    Real gamma,
    const Real *LambdaP,
    const Real *W,
    int fn,
    Real *H
);

int TestSymmetricAverage_GPU(
    int averager,
    Real gamma,
    const Real *UL,
    const Real *UR,
    Real *Uo
);

#ifdef __cplusplus
}
#endif

#endif