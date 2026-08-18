#include <cuda_runtime.h>
#include "commons.h"
#include "gpu_state.h"

/* NOTE: State variables (d_U, etc.) have been moved to gpu_fluid_dynamics.cu 
   to centralize memory management for this port.
   
   This file is intentionally left empty of definitions to prevent 
   linker errors (multiple definition of GPU_Get_U_TN).
*/