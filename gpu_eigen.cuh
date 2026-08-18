#ifndef GPU_EIGEN_CUH
#define GPU_EIGEN_CUH

#include "commons.h"

#ifdef __cplusplus
extern "C" {
#endif

int LaunchEigenvalueGPU(
    int N,
    int s,
    const Real* Uo_h,
    Real* Lambda_h
);

#ifdef __cplusplus
}
#endif

#endif

