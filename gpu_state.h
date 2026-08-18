#ifndef ARTRACFD_GPU_STATE_H_
#define ARTRACFD_GPU_STATE_H_

#include "commons.h"

#ifdef __cplusplus
extern "C" {
#endif

int GPUFluid_Init(int Nnodes);
void GPUFluid_Finalize(void);

int GPUFluid_UploadU(int Nnodes, const Real *U_TO_h, const Real *U_TN_h, const Real *U_TM_h);
int GPUFluid_DownloadU_TN(int Nnodes, Real *U_TN_h);

/* Pointer Getters */
Real* GPU_Get_U_TO(void);
Real* GPU_Get_U_TN(void);
Real* GPU_Get_U_TM(void);
Real* GPU_Get_Fhat(void);
Real* GPU_Get_Fvhat(void);
Real* GPU_Get_Phi(void);

#ifdef __cplusplus
}
#endif

#endif