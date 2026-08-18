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
#include "postprocess.h"
#include <stdio.h>   /* standard library for input and output */
#include <stdlib.h>  /* dynamic memory allocation and exit */
#include "commons.h"
#include "gpu_fluid_dynamics.h"  /* GPU finalize */

/****************************************************************************
 * Static Function Declarations
 ****************************************************************************/
static void ReleaseProgramMemory(Time *, Space *, Model *);

/****************************************************************************
 * Function Definitions
 ****************************************************************************/
int Postprocess(Time *time, Space *space, Model *model)
{
    ShowInfo("Postprocessing...\n");
    ShowInfo("  releasing memory...\n");

    /* ---- GPU cleanup (only on rank 0 if MPI) ---- */
    int do_gpu_cleanup = 1;
#ifdef MPI_ENABLED
    {
        extern int ARTRACFD_RUNMODE;
        if (ARTRACFD_RUNMODE == 2) {
            int mpi_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            do_gpu_cleanup = (mpi_rank == 0);
        }
    }
#endif
    if (do_gpu_cleanup) {
        GPUFluid_Finalize();
    }
    /* --------------------------------------------- */

    ReleaseProgramMemory(time, space, model);
    ShowInfo("  computing finished, successfully exit.\n");
    ShowInfo("Session");
    return 0;
}

static void ReleaseProgramMemory(Time *time, Space *space, Model *model)
{
    (void)time; /* time is only used for retrieving storage, keep signature */

    /* geometry related */
    Geometry *const geo = &(space->geo);
    Polyhedron *poly = NULL;
    for (int n = geo->sphN; n < geo->totN; ++n) {
        poly = geo->poly + n;
        RetrieveStorage(poly->f);
        RetrieveStorage(poly->Nf);
        RetrieveStorage(poly->e);
        RetrieveStorage(poly->Ne);
        RetrieveStorage(poly->v);
        RetrieveStorage(poly->Nv);
    }
    if (space->geo.poly != NULL) {
        free(space->geo.poly);
        space->geo.poly = NULL;
    }
    RetrieveStorage(geo->col);

    /* space related */
    Partition *const part = &(space->part);
    RetrieveStorage(part->typeBC);
    RetrieveStorage(part->N);
    RetrieveStorage(part->varBC);
    RetrieveStorage(part->typeIC);
    RetrieveStorage(part->posIC);
    RetrieveStorage(part->varIC);
    RetrieveStorage(space->node);

    /* time related */
    RetrieveStorage(time->lp);
    RetrieveStorage(time->pp);

    /* model related */
    RetrieveStorage(model->mat);
    return;
}
/* a good practice: end file with a newline */
