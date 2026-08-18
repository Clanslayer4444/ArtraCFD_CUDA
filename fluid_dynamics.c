/****************************************************************************
 *                              ArtraCFD                                    *
 *                          <By Huangrui Mo>                                *
 * Copyright (C) Huangrui Mo <huangrui.mo@gmail.com>                        *
 * This file is part of ArtraCFD.                                           *
 * ArtraCFD is free software: you can redistribute it and/or modify it      *
 * under the terms of the GNU General Public License as published by        *
 * the Free Software Foundation, either version 3 of the License, or        *
 * (at your option) any later version.                                      *
 ****************************************************************************/
/****************************************************************************
 * Required Header Files
 ****************************************************************************/
#include "fluid_dynamics.h"
#include "convective_flux.h"
#include "diffusive_flux.h"
#include "source_term.h"
#include "boundary_treatment.h"
#include "cfd_commons.h"
#include "commons.h"

#ifdef MPI_ENABLED
#include <mpi.h>
#endif

typedef void (*TimeIntegrator)(const Real, const int, Space *, const Model *);

static void DiscretizeTime(const Real, const int, Space *, const Model *);
static void RungeKutta2(const Real, const int, Space *, const Model *);
static void RungeKutta3(const Real, const int, Space *, const Model *);
static void RungeKutta3_MPI(const Real, const int, Space *, const Model *);
static void LLLU(const Real, const Real, const Real, const int,
        const int, const int, const int, Space *, const Model *);
static void LU(const Real [restrict], const Real [restrict],
        const Real [restrict], const Real [restrict], Real [restrict]);
static void SolveOperator(const int, const int, const Real, const Real,
        const Real [restrict], const Real [restrict], Real [restrict], const Real,
        const Real [restrict]);

static TimeIntegrator IntegrateTime[2] = {
    RungeKutta2,
    RungeKutta3};

void EvolveFluidDynamics(const Real dt, Space *space, const Model *model)
{
    if (0 != model->sState) {
        DiscretizeTime(0.5 * dt, PHI, space, model);
    }
    switch (model->multidim) {
        case OPTSPLIT:
            switch (space->part.collapse) {
                case COLLAPSEN:
                    DiscretizeTime(0.5 * dt, Z, space, model);
                    DiscretizeTime(0.5 * dt, Y, space, model);
                    DiscretizeTime(0.5 * dt, X, space, model);
                    DiscretizeTime(0.5 * dt, X, space, model);
                    DiscretizeTime(0.5 * dt, Y, space, model);
                    DiscretizeTime(0.5 * dt, Z, space, model);
                    break;
                case COLLAPSEX:
                    DiscretizeTime(0.5 * dt, Z, space, model);
                    DiscretizeTime(0.5 * dt, Y, space, model);
                    DiscretizeTime(0.5 * dt, Y, space, model);
                    DiscretizeTime(0.5 * dt, Z, space, model);
                    break;
                case COLLAPSEY:
                    DiscretizeTime(0.5 * dt, Z, space, model);
                    DiscretizeTime(0.5 * dt, X, space, model);
                    DiscretizeTime(0.5 * dt, X, space, model);
                    DiscretizeTime(0.5 * dt, Z, space, model);
                    break;
                case COLLAPSEZ:
                    DiscretizeTime(0.5 * dt, Y, space, model);
                    DiscretizeTime(0.5 * dt, X, space, model);
                    DiscretizeTime(0.5 * dt, X, space, model);
                    DiscretizeTime(0.5 * dt, Y, space, model);
                    break;
                case COLLAPSEXY:
                    DiscretizeTime(0.5 * dt, Z, space, model);
                    DiscretizeTime(0.5 * dt, Z, space, model);
                    break;
                case COLLAPSEXZ:
                    DiscretizeTime(0.5 * dt, Y, space, model);
                    DiscretizeTime(0.5 * dt, Y, space, model);
                    break;
                case COLLAPSEYZ:
                    DiscretizeTime(0.5 * dt, X, space, model);
                    DiscretizeTime(0.5 * dt, X, space, model);
                    break;
                default:
                    break;
            }
            break;
        case OPTBYOPT:
            DiscretizeTime(0.5 * dt, DIMS, space, model);
            DiscretizeTime(0.5 * dt, DIMS, space, model);
            break;
        default:
            break;
    }
    if (0 != model->sState) {
        DiscretizeTime(0.5 * dt, PHI, space, model);
    }
    return;
}

static void DiscretizeTime(const Real dt, const int s, Space *space, const Model *model)
{
    IntegrateTime[model->tScheme](dt, s, space, model);
    return;
}

/* ------------------------------------------------------------
 * Run Mode dispatch
 * ------------------------------------------------------------ */
extern int ARTRACFD_RUNMODE;

#ifdef CUDA_ENABLED
extern void LaunchRK3FullStep(int s, int nx, int ny, int nz, Real dt, Real ds, 
    Real gamma, Real gasR, Real refMu, Real refT, Real Prandtl, 
    const Partition *part, const Model *model);
#endif

static void RungeKutta2(const Real dt, const int s, Space *space, const Model *model)
{
    LLLU(dt, 0.0, 1.0, TO, TO, TN, s, space, model);
    TreatBoundary(TN, space, model);
    LLLU(dt, 1.0/2.0, 1.0/2.0, TO, TN, TO, s, space, model);
    TreatBoundary(TO, space, model);
}

/*
 * RK3 integration with MPI+GPU support.
 *
 * Mode dispatch:
 *   ARTRACFD_RUNMODE == 1 : GPU-only (single process, full domain on GPU)
 *   ARTRACFD_RUNMODE == 2 : MPI+GPU (rank 0 drives GPU, all ranks for I/O)
 *   otherwise             : CPU-only (original serial logic)
 */
static void RungeKutta3(const Real dt, const int s, Space *space, const Model *model)
{
    const int is_mpi_gpu = (ARTRACFD_RUNMODE == 2);

#ifdef MPI_ENABLED
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#else
    const int mpi_rank = 0;
#endif

    if (is_mpi_gpu) {
        /* MPI+GPU mode: only rank 0 runs the GPU kernels */
#ifdef MPI_ENABLED
        if (mpi_rank == 0) {
#endif
#ifdef CUDA_ENABLED
            const Partition *part = &(space->part);
            int full_nx = part->ns[PAL][X][MAX] - part->ns[PAL][X][MIN];
            int full_ny = part->ns[PAL][Y][MAX] - part->ns[PAL][Y][MIN];
            int full_nz = part->ns[PAL][Z][MAX] - part->ns[PAL][Z][MIN];

            LaunchRK3FullStep(s, full_nx, full_ny, full_nz, dt, part->dd[s], 
                model->gamma, model->gasR, model->refMu, model->refT, 0.71, part, model);
#endif
#ifdef MPI_ENABLED
        }
        /* Synchronize all ranks after GPU computation */
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        return;
    }

#ifdef CUDA_ENABLED
    if (ARTRACFD_RUNMODE == 1) {
        const Partition *part = &(space->part);
        int full_nx = part->ns[PAL][X][MAX] - part->ns[PAL][X][MIN];
        int full_ny = part->ns[PAL][Y][MAX] - part->ns[PAL][Y][MIN];
        int full_nz = part->ns[PAL][Z][MAX] - part->ns[PAL][Z][MIN];

        LaunchRK3FullStep(s, full_nx, full_ny, full_nz, dt, part->dd[s], 
            model->gamma, model->gasR, model->refMu, model->refT, 0.71, part, model);
        return;
    }
#endif

    /* CPU-only RK3 */
    LLLU(dt, 0.0, 1.0, TO, TO, TN, s, space, model);
    TreatBoundary(TN, space, model);
    LLLU(dt, 3.0/4.0, 1.0/4.0, TO, TN, TM, s, space, model);
    TreatBoundary(TM, space, model);
    LLLU(dt, 1.0/3.0, 2.0/3.0, TO, TM, TO, s, space, model);
    TreatBoundary(TO, space, model);
}

static void LLLU(const Real dt, const Real coeA, const Real coeB, const int to,
        const int tn, const int tm, const int p, Space *space, const Model *model)
{
    const Partition *const part = &(space->part);
    Node *const node = space->node;
    int idx = 0; 
    int i = 0, j = 0, k = 0;
    const int h[DIMS][DIMS] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    Real RHS[5][DIMU] = {{0.0}}; 
    Real *restrict FhatR = RHS[0]; 
    Real *restrict FhatL = RHS[1]; 
    Real *restrict FvhatR = RHS[2]; 
    Real *restrict FvhatL = RHS[3]; 
    Real *restrict Phi = RHS[4]; 
    Real *temp = NULL;
    const IntVec partn = {part->n[X], part->n[Y], part->n[Z]};
    const RealVec dd = {part->dd[X], part->dd[Y], part->dd[Z]};
    const RealVec r = {dt * dd[X], dt * dd[Y], dt * dd[Z]};
    int s = 0, sN = 0; 
    switch (p) {
        case PHI: 
            s = 0; sN = s + 1;
            break;
        case DIMS: 
            s = 0; sN = DIMS;
            break;
        default: 
            s = p; sN = s + 1;
            break;
    }
    for (; s < sN; ++s) {
        for (int ks = part->np[s][Z][MIN]; ks < part->np[s][Z][MAX]; ++ks) {
            for (int js = part->np[s][Y][MIN]; js < part->np[s][Y][MAX]; ++js) {
                for (int is = part->np[s][X][MIN], state = 0; is < part->np[s][X][MAX]; ++is) {
                    switch (s) {
                        case X:
                            i = is; j = js; k = ks;
                            break;
                        case Y:
                            i = js; j = is; k = ks;
                            break;
                        case Z:
                            i = js; j = ks; k = is;
                            break;
                        default:
                            break;
                    }
                    idx = IndexNode(k, j, i, partn[Y], partn[X]);
                    if (0 != node[idx].did) {
                        state = 0; 
                        continue;
                    }
                    switch (p) {
                        case PHI:
                            ComputePhi(tn, k, j, i, partn, node, model, Phi);
                            SolveOperator(OPTSPLIT, s, coeA, coeB, node[idx].U[to], node[idx].U[tn], node[idx].U[tm], dt, Phi);
                            continue;
                        default:
                            break;
                    }
                    switch (state) {
                        case 1: 
                            temp = FhatL;
                            FhatL = FhatR;
                            FhatR = temp;
                            temp = FvhatL;
                            FvhatL = FvhatR;
                            FvhatR = temp;
                            break;
                        default: 
                            ComputeFhat(tn, s, k - h[s][Z], j - h[s][Y], i - h[s][X], partn, node, model, FhatL);
                            ComputeFvhat(tn, s, k - h[s][Z], j - h[s][Y], i - h[s][X], partn, dd, node, model, FvhatL);
                            state = 1;
                            break;
                    }
                    ComputeFhat(tn, s, k, j, i, partn, node, model, FhatR);
                    ComputeFvhat(tn, s, k, j, i, partn, dd, node, model, FvhatR);
                    LU(FhatR, FhatL, FvhatR, FvhatL, Phi);

                    SolveOperator(model->multidim, s, coeA, coeB, node[idx].U[to], node[idx].U[tn], node[idx].U[tm], r[s], Phi);
                }
            }
        }
    }
    return;
}

static void LU(const Real FhatR[restrict], const Real FhatL[restrict],
        const Real FvhatR[restrict], const Real FvhatL[restrict], Real Phi[restrict])
{
    for (int n = 0; n < DIMU; ++n) {
        Phi[n] = FhatL[n] - FhatR[n] + FvhatR[n] - FvhatL[n];
    }
    return;
}

static void SolveOperator(const int p, const int s, const Real coeA, const Real coeB,
        const Real Uo[restrict], const Real Un[restrict], Real Um[restrict], const Real r,
        const Real Phi[restrict])
{
    if ((OPTBYOPT == p) && (X != s)) {
        for (int n = 0; n < DIMU; ++n) {
            Um[n] = Um[n] + coeB * r * Phi[n];
        }
    } else {
        for (int n = 0; n < DIMU; ++n) {
            Um[n] = coeA * Uo[n] + coeB * (Un[n] + r * Phi[n]);
        }
    }

    Real gamma = 1.4;
    if (Um[0] < 1e-4) {
        Um[0] = 1e-4;
        Um[1] = 0.0; Um[2] = 0.0; Um[3] = 0.0;
    }
    Real kin_energy = 0.5 * (Um[1]*Um[1] + Um[2]*Um[2] + Um[3]*Um[3]) / Um[0];
    Real pressure = (gamma - 1.0) * (Um[4] - kin_energy);

    if (pressure < 1e-4) {
        pressure = 1e-4;
        Um[4] = pressure / (gamma - 1.0) + kin_energy; 
    }
    return;
}
/* a good practice: end file with a newline */
