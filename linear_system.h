#ifndef ARTRACFD_LINEAR_SYSTEM_H_
#define ARTRACFD_LINEAR_SYSTEM_H_

#include "commons.h"

// Guarantee RESTRICT always exists
#ifndef RESTRICT
  #ifdef __CUDACC__
    #define RESTRICT __restrict__
  #else
    #define RESTRICT restrict
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Use parentheses to keep NVCC ....satisfied..
extern int SolveLinearSystem(const int n, Real (*RESTRICT A),
        const int m, Real (*RESTRICT X), Real (*RESTRICT B));

#ifdef __cplusplus 
}
#endif

#endif 
