/****************************************************************************
 *                              ArtraCFD                                    *
 *                          <By Huangrui Mo>                                *
 *                                                                          *
 * MPI Interface Layer Implementation                                       *
 *   Provides distributed computing support for Multi-CPU + 1 GPU mode.     *
 *   Rank 0 owns the GPU. All ranks participate in domain decomposition.    *
 ****************************************************************************/
#include "mpi_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>  /* gethostname() */

#ifdef MPI_ENABLED

/* ---------------------------------------------------------------------------
 * Initialize MPI environment
 * ------------------------------------------------------------------------- */
int MPI_InitSolver(int *argc, char ***argv, int *rank, int *nprocs)
{
    int ret = MPI_Init(argc, argv);
    if (ret != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init failed with error code %d\n", ret);
        return ret;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, nprocs);

    return MPI_SUCCESS;
}

/* ---------------------------------------------------------------------------
 * Finalize MPI environment
 * ------------------------------------------------------------------------- */
void MPI_FinalizeSolver(void)
{
    MPI_Finalize();
}

/* ---------------------------------------------------------------------------
 * Compute subdomain index ranges for a given rank.
 * Decomposes along the Y dimension (good for 2D/3D structured grids).
 * Each rank gets a slab of the domain: continuous in X, sliced in Y.
 *
 * For nprocs > ny, some ranks will have zero-width subdomains.
 * ------------------------------------------------------------------------- */
void MPI_ComputeSubdomain(
    int rank, int nprocs,
    int nx, int ny, int nz,
    int ng,
    int local_box[6])
{
    /* Use a simple slab decomposition along Y */
    int ny_local = ny / nprocs;
    int ny_remainder = ny % nprocs;

    int y_start, y_end;
    if (rank < ny_remainder) {
        y_start = rank * (ny_local + 1);
        y_end = y_start + (ny_local + 1);
    } else {
        y_start = ny_remainder * (ny_local + 1) + (rank - ny_remainder) * ny_local;
        y_end = y_start + ny_local;
    }

    /* Clamp to global bounds */
    if (y_start < 0) y_start = 0;
    if (y_end > ny) y_end = ny;

    /* Extend with ghost layers */
    int y_local_min = y_start - ng;
    int y_local_max = y_end + ng;
    if (y_local_min < 0) y_local_min = 0;
    if (y_local_max > ny) y_local_max = ny;

    local_box[0] = 0;              /* x_min */
    local_box[1] = nx;             /* x_max */
    local_box[2] = y_local_min;    /* y_min (with ghosts) */
    local_box[3] = y_local_max;    /* y_max (with ghosts) */
    local_box[4] = 0;              /* z_min */
    local_box[5] = nz;             /* z_max */
}

/* ---------------------------------------------------------------------------
 * Halo Exchange for 5-component fluid state array
 *
 * Uses MPI_Sendrecv for neighbor-to-neighbor communication.
 * Each rank sends its boundary rows to the left/right neighbors
 * and receives ghost rows from them.
 *
 * State array layout: U[5 * (k * ny * nx + j * nx + i) + v]
 * ------------------------------------------------------------------------- */
void MPI_ExchangeHalo(
    Real *U_local,
    const int local_box[6],
    int nx, int ny, int nz,
    int ng,
    int rank, int nprocs,
    MPI_Comm comm)
{
    if (nprocs <= 1) return; /* no exchange needed for single process */

    const int x_min = local_box[0];
    const int x_max = local_box[1];
    const int y_min = local_box[2];
    const int y_max = local_box[3];
    const int z_min = local_box[4];
    const int z_max = local_box[5];

    const int local_nx = x_max - x_min;
    const int local_ny = y_max - y_min;
    const int local_nz = z_max - z_min;

    /* Determine neighbors */
    int left_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int right_neighbor = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    /* Compute sizes for the halo regions (plane of data in Y-Z) */
    int plane_size = local_nx * local_nz * 5; /* 5 components per node */

    /* Allocate send/recv buffers */
    Real *send_left = NULL, *recv_left = NULL;
    Real *send_right = NULL, *recv_right = NULL;

    /* --- Exchange in Y direction --- */

    /* Send to left neighbor: our first interior Y layers */
    /* Receive from left neighbor: their last Y layers (our ghost) */
    if (left_neighbor != MPI_PROC_NULL) {
        send_left = (Real *)malloc(ng * plane_size * sizeof(Real));
        recv_left = (Real *)malloc(ng * plane_size * sizeof(Real));

        /* Pack: our first ng interior layers (y = y_local_min + ng to y_local_min + 2*ng - 1) */
        for (int g = 0; g < ng; ++g) {
            int src_y = y_min + ng + g;
            for (int k = z_min; k < z_max; ++k) {
                for (int i = x_min; i < x_max; ++i) {
                    int src_idx = (k * local_ny + (src_y - y_min)) * local_nx + (i - x_min);
                    int dst_idx = (g * local_nz + (k - z_min)) * local_nx + (i - x_min);
                    for (int v = 0; v < 5; ++v) {
                        send_left[dst_idx * 5 + v] = U_local[src_idx * 5 + v];
                    }
                }
            }
        }

        MPI_Sendrecv(
            send_left, ng * plane_size, MPI_DOUBLE, left_neighbor, 0,
            recv_left, ng * plane_size, MPI_DOUBLE, left_neighbor, 0,
            comm, MPI_STATUS_IGNORE);

        /* Unpack into ghost region (y = y_min to y_min + ng - 1) */
        for (int g = 0; g < ng; ++g) {
            int dst_y = y_min + g;
            for (int k = z_min; k < z_max; ++k) {
                for (int i = x_min; i < x_max; ++i) {
                    int dst_idx = (k * local_ny + (dst_y - y_min)) * local_nx + (i - x_min);
                    int src_idx = (g * local_nz + (k - z_min)) * local_nx + (i - x_min);
                    for (int v = 0; v < 5; ++v) {
                        U_local[dst_idx * 5 + v] = recv_left[src_idx * 5 + v];
                    }
                }
            }
        }

        free(send_left);
        free(recv_left);
    }

    /* Send to right neighbor: our last interior Y layers */
    /* Receive from right neighbor: their first Y layers (our ghost) */
    if (right_neighbor != MPI_PROC_NULL) {
        send_right = (Real *)malloc(ng * plane_size * sizeof(Real));
        recv_right = (Real *)malloc(ng * plane_size * sizeof(Real));

        /* Pack: our last ng interior layers (y = y_max - 2*ng to y_max - ng - 1) */
        for (int g = 0; g < ng; ++g) {
            int src_y = y_max - ng - 1 - g;
            for (int k = z_min; k < z_max; ++k) {
                for (int i = x_min; i < x_max; ++i) {
                    int src_idx = (k * local_ny + (src_y - y_min)) * local_nx + (i - x_min);
                    int dst_idx = (g * local_nz + (k - z_min)) * local_nx + (i - x_min);
                    for (int v = 0; v < 5; ++v) {
                        send_right[dst_idx * 5 + v] = U_local[src_idx * 5 + v];
                    }
                }
            }
        }

        MPI_Sendrecv(
            send_right, ng * plane_size, MPI_DOUBLE, right_neighbor, 0,
            recv_right, ng * plane_size, MPI_DOUBLE, right_neighbor, 0,
            comm, MPI_STATUS_IGNORE);

        /* Unpack into ghost region (y = y_max - ng to y_max - 1) */
        for (int g = 0; g < ng; ++g) {
            int dst_y = y_max - ng + g;
            for (int k = z_min; k < z_max; ++k) {
                for (int i = x_min; i < x_max; ++i) {
                    int dst_idx = (k * local_ny + (dst_y - y_min)) * local_nx + (i - x_min);
                    int src_idx = (g * local_nz + (k - z_min)) * local_nx + (i - x_min);
                    for (int v = 0; v < 5; ++v) {
                        U_local[dst_idx * 5 + v] = recv_right[src_idx * 5 + v];
                    }
                }
            }
        }

        free(send_right);
        free(recv_right);
    }
}

/* ---------------------------------------------------------------------------
 * Broadcast boundary condition arrays from rank 0 to all ranks
 * ------------------------------------------------------------------------- */
void MPI_BcastBoundaryData(
    int *typeBC, Real *varBC,
    int rank, MPI_Comm comm)
{
    /* Broadcast boundary types (6 boundaries) */
    MPI_Bcast(typeBC, 6, MPI_INT, MPI_RANK_GPU_OWNER, comm);

    /* Broadcast boundary values (6 boundaries × 6 values each) */
    MPI_Bcast(varBC, 36, MPI_DOUBLE, MPI_RANK_GPU_OWNER, comm);
}

/* ---------------------------------------------------------------------------
 * Gather full solution to rank 0
 * Uses MPI_Gatherv for variable-size subdomains.
 * ------------------------------------------------------------------------- */
void MPI_GatherSolution(
    const Real *U_local, int local_nnodes,
    Real *U_global,
    int total_nnodes,
    int rank, int nprocs, MPI_Comm comm)
{
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == MPI_RANK_GPU_OWNER) {
        recvcounts = (int *)malloc(nprocs * sizeof(int));
        displs = (int *)malloc(nprocs * sizeof(int));
    }

    /* Gather sizes first */
    MPI_Gather(&local_nnodes, 1, MPI_INT,
               recvcounts, 1, MPI_INT,
               MPI_RANK_GPU_OWNER, comm);

    if (rank == MPI_RANK_GPU_OWNER) {
        /* Compute displacements */
        displs[0] = 0;
        for (int i = 1; i < nprocs; ++i) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }

    /* Gather the actual data */
    MPI_Gatherv(U_local, local_nnodes * 5, MPI_DOUBLE,
                U_global, recvcounts, displs, MPI_DOUBLE,
                MPI_RANK_GPU_OWNER, comm);

    if (rank == MPI_RANK_GPU_OWNER) {
        free(recvcounts);
        free(displs);
    }
}

/* ---------------------------------------------------------------------------
 * Print rank information
 * ------------------------------------------------------------------------- */
void MPI_PrintRankInfo(int rank, int nprocs)
{
    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    printf("  >> MPI Rank %d/%d on %s\n", rank, nprocs, hostname);
    fflush(stdout);

    /* Barrier to avoid interleaved output */
    MPI_Barrier(MPI_COMM_WORLD);
}

#endif /* MPI_ENABLED */
