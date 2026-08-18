#ifndef GPU_IBM_CUH
#define GPU_IBM_CUH

#include "commons.h"

#define MAX_STENCIL 125

typedef struct {
    int ghost_idx;
    int num_neighbors;
    int neighbor_indices[MAX_STENCIL];
    int padding1; // FORCES 8-BYTE ALIGNMENT: 4+4+500+4 = 512 bytes

    Real weights[MAX_STENCIL];
    Real weight_sum;

    /* Mirroring Data (For nodes close to the wall) */
    int is_mirror;
    int padding2; // FORCES 8-BYTE ALIGNMENT: 4+4 = 8 bytes

    Real N[3];       /* Surface Normal */
    Real pO[3];      /* Surface Intersect Point */
    Real poly_V[3];  /* Polygon Linear Velocity */
    Real poly_W[3];  /* Polygon Angular Velocity */
    Real poly_O[3];  /* Polygon Origin */
    Real poly_T;     /* Polygon Temperature */
    Real poly_cf;    /* Friction Coefficient Flag */
} IBM_Stencil;

#endif