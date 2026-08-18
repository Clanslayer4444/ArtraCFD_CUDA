#ifndef ARTRACFD_GPU_EIGEN_H_
#define ARTRACFD_GPU_EIGEN_H_

#include "commons.h"

#ifdef __cplusplus
extern "C" {
#endif

int LaunchEigenTestGPU(
    int N,
    int s,
    Real gamma,
    const Real *Uo_h,
    Real *Lambda_h,
    Real *L_h,
    Real *R_h
);

int LaunchEigenSplitTestGPU(
    int splitter,
    const Real *Lambda,
    Real *LambdaP,
    Real *LambdaN
);

#ifdef __cplusplus
}
#endif

#endif