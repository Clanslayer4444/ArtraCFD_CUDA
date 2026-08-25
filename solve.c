#include "solve.h"
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include "initialization.h"
#include "fluid_dynamics.h"
#include "solid_dynamics.h"
#include "data_stream.h"
#include "timer.h"
#include "cfd_commons.h"
#include "commons.h"
#include "gpu_fluid_dynamics.h"

#ifdef MPI_ENABLED
#include <mpi.h>
#include "mpi_interface.h"
#endif

static void EvolveSolution(Time *time, Space *space, const Model *model);
static Real ComputeTimeStep(const Time *time, const Space *space, const Model *model);

#ifdef CUDA_ENABLED
Real *h_flat_U_buffer = NULL;   /* Host-side flat buffer for I/O transfers */
extern Real *d_U;               /* Device pointer: single source of truth on GPU */
#endif

int Solve(Time *time, Space *space, const Model *model) {
    /* Determine MPI rank for distributed I/O */
    int mpi_rank = 0;
    int mpi_nprocs = 1;
#ifdef MPI_ENABLED
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_nprocs);
#endif

    /* Only rank 0 initializes and solves */
    if (mpi_rank == 0) {
        ShowInfo("Solving...\n  initializing...\n");
        InitializeComputeDomain(time, space, model);
    }

#ifdef MPI_ENABLED
    /* Broadcast init state to all ranks */
    if (ARTRACFD_RUNMODE == 2 && mpi_nprocs > 1) {
        /* Broadcast time parameters */
        MPI_Bcast(&time->now, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&time->stepC, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
#endif

    #ifdef CUDA_ENABLED
    extern int ARTRACFD_RUNMODE;
    if (ARTRACFD_RUNMODE == 1 || (ARTRACFD_RUNMODE == 2 && mpi_rank == 0)) {
        extern void BuildIBMMap(Space *space, const Model *model);
        extern void InitIBM_GPU(void *h_map, int num_stencils);
        extern void SendGPUBoundaryData(const Real *h_varBC);
        extern void SendGPUDomainData(const int *h_did, int Nnodes);
        extern void LaunchBoundaryKernel(int nx, int ny, int nz, const Partition *part, const Model *model);
        
        extern int GPUFluid_Init(int Nnodes);
        extern void *h_ibm_map;
        extern int num_ibm_stencils;

        ShowInfo("  building GPU Immersed Boundary Map...\n");
        BuildIBMMap(space, model);
        InitIBM_GPU(h_ibm_map, num_ibm_stencils);

        int full_nx = space->part.ns[PAL][X][MAX] - space->part.ns[PAL][X][MIN];
        int full_ny = space->part.ns[PAL][Y][MAX] - space->part.ns[PAL][Y][MIN];
        int full_nz = space->part.ns[PAL][Z][MAX] - space->part.ns[PAL][Z][MIN];
        int total_nodes = full_nx * full_ny * full_nz;

        GPUFluid_Init(total_nodes);
        SendGPUBoundaryData((Real*)space->part.varBC);

        /* Upload initial domain tags (did array) to GPU */
        int *h_flat_did = (int *)malloc(total_nodes * sizeof(int));
        for (int i = 0; i < total_nodes; ++i) h_flat_did[i] = space->node[i].did;
        SendGPUDomainData(h_flat_did, total_nodes);
        free(h_flat_did);

        /* Upload initial fluid state to GPU (one-time init) */
        h_flat_U_buffer = (Real *)malloc(total_nodes * 5 * sizeof(Real));

        /* PHASE 1: Allocate GPU memory and upload initial state */
        for (int i = 0; i < total_nodes; ++i) {
            for (int v = 0; v < 5; ++v) {
                h_flat_U_buffer[i * 5 + v] = space->node[i].U[TO][v];
            }
        }
        SendGPUFluidData(h_flat_U_buffer, total_nodes);
        ShowInfo("  >> GPU initialized: fluid state uploaded to VRAM\n");
        free(h_flat_U_buffer);  /* Free host buffer — we reconstruct on I/O */
        h_flat_U_buffer = NULL;

        /* Apply boundary and IBM on GPU to match CPU init state */
        LaunchBoundaryKernel(full_nx, full_ny, full_nz, &space->part, model);

        ShowInfo("  >> GPU init complete.\n");
    }
    #endif

    ShowInfo("  time marching...\n");
    EvolveSolution(time, space, model);
    ShowInfo("Session");

    #ifdef CUDA_ENABLED
    if (ARTRACFD_RUNMODE == 1) {
        extern void GPUFluid_Finalize(void);
        GPUFluid_Finalize();
        if (h_flat_U_buffer) free(h_flat_U_buffer);
    }
    #endif

    return 0;
}

/*
 * Pull GPU fluid state back to CPU memory.
 * Only called when I/O output is needed.
 */
#ifdef CUDA_ENABLED
static void SyncGPUToCPU(const Space *space, int total_nodes) {
    extern void FetchGPUFluidData(Real *h_U, int Nnodes);

    /* Allocate buffer on first use */
    if (h_flat_U_buffer == NULL) {
        h_flat_U_buffer = (Real *)malloc(total_nodes * 5 * sizeof(Real));
    }

    FetchGPUFluidData(h_flat_U_buffer, total_nodes);
    for (int i = 0; i < total_nodes; ++i) {
        for (int v = 0; v < 5; ++v) {
            space->node[i].U[TO][v] = h_flat_U_buffer[i * 5 + v];
        }
    }
}
#endif

static void EvolveSolution(Time *time, Space *space, const Model *model) {
    Real dt = time->end - time->now;
    if (0.0 >= dt) { ShowWarning("  time.now >= time.end"); return; }

    int is3D = (space->part.n[Z] > 1);
    Timer tm;
    const Real dtData[NPROBE] = {time->end / (Real)(time->dataW[PROPT]), time->end / (Real)(time->dataW[PROLN]), time->end / (Real)(time->dataW[PROCV]), time->end / (Real)(time->dataW[PROFC]), time->end / (Real)(time->dataW[PROSD])};
    Real rcData[NPROBE] = {0.0};
    const Real tmInt = (INT_MAX == time->dataW[PROSD]) ? time->end : dtData[PROSD];
    Real rcInt = 0.0;

    int total_nX = space->part.ns[PAL][X][MAX] - space->part.ns[PAL][X][MIN];
    int total_nY = space->part.ns[PAL][Y][MAX] - space->part.ns[PAL][Y][MIN];
    int total_nZ = space->part.ns[PAL][Z][MAX] - space->part.ns[PAL][Z][MIN];
    int total_nodes = total_nX * total_nY * total_nZ;

    #ifdef CUDA_ENABLED
    extern int ARTRACFD_RUNMODE;
    const int gpu_mode = ARTRACFD_RUNMODE;
    #else
    const int gpu_mode = 0;
    #endif

    while ((time->now < time->end) && (time->stepC < time->stepN)) {
        ++(time->stepC);

        dt = ComputeTimeStep(time, space, model);
        if (dt > 0.01) dt = 0.01;

        if (rcInt + dt > tmInt) { dt = tmInt - rcInt; rcInt = 0.0; } else { rcInt = rcInt + dt; }
        time->now = time->now + dt;
        if (time->now > time->end) { dt = time->end - (time->now - dt); time->now = time->end; }

        ShowInfo("\nstep=%d; time=%.6g; remain=%.6g; dt=%.6g;\n", time->stepC, time->now, time->end - time->now, dt);
        TickTime(&tm);

        /* --- Solid Dynamics (always on CPU, needs synced fluid state) --- */
        if (0 != model->psi) {
            #ifdef CUDA_ENABLED
            if (gpu_mode) {
                /* Pull fluid state from GPU before solid dynamics integration */
                SyncGPUToCPU(space, total_nodes);
            }
            #endif
            EvolveSolidDynamics(time->now, 0.5 * dt, space, model);
        }

        /* --- Fluid Dynamics --- */
        EvolveFluidDynamics(dt, space, model);

        /* --- GPU is now authoritative for fluid state --- */

        /* --- Solid Dynamics post-step (also needs synced state) --- */
        if (0 != model->psi) {
            #ifdef CUDA_ENABLED
            if (gpu_mode) {
                SyncGPUToCPU(space, total_nodes);
            }
            #endif
            EvolveSolidDynamics(time->now, 0.5 * dt, space, model);
        }

        ShowInfo("  elapsed: %.6gs\n", TockTime(&tm));

        /* --- I/O and probe output --- */
        int io_needed = 0;
        for (int n = 0; n < NPROBE; ++n) {
            rcData[n] = rcData[n] + dt;
            if ((rcData[n] >= dtData[n]) || (time->now == time->end) || (time->stepC == time->stepN)) {
                io_needed = 1;
            }
        }

        if (io_needed) {
            #ifdef CUDA_ENABLED
            if (gpu_mode) {
                /* Sync GPU → CPU for I/O output */
                SyncGPUToCPU(space, total_nodes);
            }
            #endif

            for (int n = 0; n < NPROBE; ++n) {
                if ((rcData[n] >= dtData[n]) || (time->now == time->end) || (time->stepC == time->stepN)) {
                    if (!is3D && (n == PROFC || n == PROCV)) { rcData[n] = 0.0; continue; }
                    if (PROFC == n) IntegrateSurfaceForce(space, model);
                    if (PROSD == n) {
                        ShowInfo("  writing data...\n");
                        ++(time->dataC);
                    }
                    WriteData(n, time, space, model);
                    rcData[n] = 0.0;
                }
            }
        }
    }
}

static Real ComputeTimeStep(const Time *time, const Space *space, const Model *model) {
    const Partition *const part = &(space->part);
    const Node *const node = space->node;
    const Geometry *const geo = &(space->geo);
    const Polyhedron *poly = NULL;
    const Real *RESTRICT U = NULL;
    Real Uo[DIMUo] = {0.0};
    int idx = 0;
    Real c = 0.0;
    RealVec V = {0.0};
    RealVec Vmax = {0.0};

    for (int n = 0; n < geo->totN; ++n) {
        poly = geo->poly + n;
        V[X] = fabs(poly->V[TO][X]) + MaxReal(fabs(poly->W[TO][Y]), fabs(poly->W[TO][Z])) * poly->r;
        V[Y] = fabs(poly->V[TO][Y]) + MaxReal(fabs(poly->W[TO][Z]), fabs(poly->W[TO][X])) * poly->r;
        V[Z] = fabs(poly->V[TO][Z]) + MaxReal(fabs(poly->W[TO][X]), fabs(poly->W[TO][Y])) * poly->r;
        for (int s = 0; s < DIMS; ++s) { if (Vmax[s] < V[s]) Vmax[s] = V[s]; }
    }

    int total_nY = part->ns[PAL][Y][MAX] - part->ns[PAL][Y][MIN];
    int total_nX = part->ns[PAL][X][MAX] - part->ns[PAL][X][MIN];
    int total_nZ = part->ns[PAL][Z][MAX] - part->ns[PAL][Z][MIN];

    extern int ARTRACFD_RUNMODE;
    if (ARTRACFD_RUNMODE == 1) {
        extern void LaunchComputeCFL_GPU(
            int nx, int ny, int nz,
            int xmin, int xmax, int ymin, int ymax, int zmin, int zmax,
            Real gamma, Real gasR, Real *Vmax_out);

        Real Vmax_gpu[3] = {0.0, 0.0, 0.0};
        LaunchComputeCFL_GPU(
            total_nX, total_nY, total_nZ,
            part->ns[PIN][X][MIN], part->ns[PIN][X][MAX],
            part->ns[PIN][Y][MIN], part->ns[PIN][Y][MAX],
            part->ns[PIN][Z][MIN], part->ns[PIN][Z][MAX],
            model->gamma, model->gasR, Vmax_gpu);

        for (int s = 0; s < DIMS; ++s) {
            if (Vmax[s] < Vmax_gpu[s]) Vmax[s] = Vmax_gpu[s];
        }
    } else {
        for (int k = part->ns[PIN][Z][MIN]; k < part->ns[PIN][Z][MAX]; ++k) {
            for (int j = part->ns[PIN][Y][MIN]; j < part->ns[PIN][Y][MAX]; ++j) {
                for (int i = part->ns[PIN][X][MIN]; i < part->ns[PIN][X][MAX]; ++i) {
                    idx = IndexNode(k, j, i, total_nY, total_nX);
                    U = node[idx].U[TO];
                    if (0 != node[idx].did) continue;
                    MapPrimitive(model->gamma, model->gasR, U, Uo);
                    c = sqrt(model->gamma * model->gasR * Uo[5]);
                    for (int s = 0; s < DIMS; ++s) {
                        V[s] = fabs(Uo[s+1]) + c;
                        if (Vmax[s] < V[s]) Vmax[s] = V[s];
                    }
                }
            }
        }
    }

    Real dtX = (Vmax[X] > 1e-6) ? part->d[X] / Vmax[X] : 1e10;
    Real dtY = (Vmax[Y] > 1e-6) ? part->d[Y] / Vmax[Y] : 1e10;
    Real dtZ = (Vmax[Z] > 1e-6) ? part->d[Z] / Vmax[Z] : 1e10;

    return time->numCFL * MinReal(dtX, MinReal(dtY, dtZ));
}
