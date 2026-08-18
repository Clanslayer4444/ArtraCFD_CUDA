/****************************************************************************
 *                              ArtraCFD                                    *
 *                          <By Huangrui Mo>                                *
 *                                                                          *
 * MPI Interface Layer                                                      *
 *   Provides distributed computing support for Multi-CPU + 1 GPU mode.     *
 *   Rank 0 owns the GPU. All ranks participate in domain decomposition.    *
 ****************************************************************************/
#ifndef ARTRACFD_MPI_INTERFACE_H_
#define ARTRACFD_MPI_INTERFACE_H_

#include "commons.h"

#ifdef MPI_ENABLED

#include <mpi.h>

/* MPI rank roles */
#define MPI_RANK_GPU_OWNER  0   /* The rank that owns the GPU */

/* =========================================================================
 * Initialization and Finalization
 * ========================================================================= */
/*
 * Initialize MPI environment.
 * Must be called before any other MPI function.
 * Returns the total number of ranks and this rank's ID.
 */
extern int MPI_InitSolver(int *argc, char ***argv, int *rank, int *nprocs);

/*
 * Finalize MPI environment.
 */
extern void MPI_FinalizeSolver(void);

/* =========================================================================
 * Domain Decomposition
 * ========================================================================= */
/*
 * Compute subdomain index ranges for a given rank.
 * Decomposes the global domain [0, nx) × [0, ny) × [0, nz) into nprocs
 * partitions along the Y dimension (good for 2D and 3D structured grids).
 *
 * Returns the local [z_min, z_max, y_min, y_max, x_min, x_max] for this rank.
 */
extern void MPI_ComputeSubdomain(
    int rank, int nprocs,
    int nx, int ny, int nz,
    int ng,                   /* number of ghost layers */
    int local_box[6]          /* output: [x_min, x_max, y_min, y_max, z_min, z_max] */
);

/* =========================================================================
 * Halo Exchange
 * ========================================================================= */
/*
 * Exchange ghost cell data for the fluid state array (flat 5-component AoS).
 * Each rank sends its boundary layers to neighbors and receives halo data.
 * Handles periodic/symmetry boundaries internally when no neighbor exists.
 */
extern void MPI_ExchangeHalo(
    Real *U_local,            /* local subdomain data [5 * local_nnodes] */
    const int local_box[6],   /* local index bounds [x_min,x_max,y_min,y_max,z_min,z_max] */
    int nx, int ny, int nz,   /* global dimensions */
    int ng,                   /* ghost layer count */
    int rank, int nprocs,
    MPI_Comm comm
);

/* =========================================================================
 * Boundary Condition Synchronization
 * ========================================================================= */
/*
 * Broadcast boundary condition arrays from rank 0 to all ranks.
 * All ranks need identical boundary conditions for their local processing.
 */
extern void MPI_BcastBoundaryData(
    int *typeBC, Real *varBC,
    int rank, MPI_Comm comm
);

/* =========================================================================
 * I/O Aggregation
 * ========================================================================= */
/*
 * Gather full solution data to rank 0 for file output.
 * rank 0 reconstructs the global array from subdomain contributions.
 */
extern void MPI_GatherSolution(
    const Real *U_local, int local_nnodes,
    Real *U_global,          /* only used on rank 0 */
    int total_nnodes,
    int rank, int nprocs, MPI_Comm comm
);

/* =========================================================================
 * Diagnostics
 * ========================================================================= */
extern void MPI_PrintRankInfo(int rank, int nprocs);

#endif /* MPI_ENABLED */

#endif /* ARTRACFD_MPI_INTERFACE_H_ */
