/****************************************************************************
 * ArtraCFD                                    *
 * <By Huangrui Mo>                                *
 * Copyright (C) Huangrui Mo <huangrui.mo@gmail.com>                        *
 * This file is part of ArtraCFD.                                           *
 * ArtraCFD is free software: you can redistribute it and/or modify it      *
 * under the terms of the GNU General Public License as published by        *
 * the Free Software Foundation, either version 3 of the License, or        *
 * (at your option) any later version.                                      *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>   /* For Wall Clock */
#include <sys/resource.h> /* For CPU/Sys Time */
#include "commons.h"
#include "program_entrance.h"
#include "preprocess.h"
#include "solve.h"
#include "postprocess.h"

#ifdef MPI_ENABLED
#include "mpi_interface.h"
#endif

/* Global Run Mode Flag */
/* 0 = Serial/CPU/OMP/MPI, 1 = GPU, 2 = MPI+GPU */
int ARTRACFD_RUNMODE = 0;

/* MPI rank info (populated in main for MPI builds) */
#ifdef MPI_ENABLED
int ARTRACFD_MPI_RANK = 0;
int ARTRACFD_MPI_NPROCS = 1;
#endif

int main(int argc, char *argv[])
{
    Control control = {.runMode = 'i', .proc = {0}};
    Time time = {0};
    Space space = {0};
    Model model = {0};
    int force_solve = 0;

    /* --- TIMING START --- */
    struct timeval start_wall, end_wall;
    gettimeofday(&start_wall, NULL);

    /* 1. Manually check flags */
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            if (strcmp(argv[i+1], "gpu") == 0) {
                ARTRACFD_RUNMODE = 1;
                force_solve = 1;
                printf(">> Mode set to GPU\n");
            } else if (strcmp(argv[i+1], "omp") == 0) {
                ARTRACFD_RUNMODE = 0;
                force_solve = 1;
                printf(">> Mode set to OpenMP\n");
            } else if (strcmp(argv[i+1], "mpi") == 0) {
                #ifdef MPI_ENABLED
                ARTRACFD_RUNMODE = 2;   /* MPI+GPU mode */
                #else
                ARTRACFD_RUNMODE = 0;   /* Fallback to serial if no MPI */
                #endif
                force_solve = 1;
                printf(">> Mode set to MPI\n");
            } else {
                ARTRACFD_RUNMODE = 0;
            }
        }
    }

    /* 1b. MPI initialization (must happen before EnterProgram) */
    #ifdef MPI_ENABLED
    if (ARTRACFD_RUNMODE == 2) {
        int mpi_rank, mpi_nprocs;
        MPI_InitSolver(&argc, &argv, &mpi_rank, &mpi_nprocs);
        ARTRACFD_MPI_RANK = mpi_rank;
        ARTRACFD_MPI_NPROCS = mpi_nprocs;
        
        if (mpi_rank == 0) {
            printf(">> MPI initialized: %d ranks\n", mpi_nprocs);
        }
        
        /* Override proc count from command line only for rank 0 */
        if (mpi_rank == 0) {
            /* Processor count is set via EnterProgram or -n flag */
        }
    }
    #endif

    /* 2. Standard Entrance */
    EnterProgram(argc, argv, &control, &space);

    /* 3. Force Solve Mode if needed (Fixes early exit for OMP/MPI/GPU) */
    if (force_solve) {
        control.runMode = 's'; 
    }

    /* 4. Run the Solver Pipeline */
    if (control.runMode == 's') { // 's' for solve
        Preprocess(&time, &space, &model);
        Solve(&time, &space, &model);
        Postprocess(&time, &space, &model);
    } 

    /* 4b. MPI Finalize */
    #ifdef MPI_ENABLED
    if (ARTRACFD_RUNMODE == 2) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_FinalizeSolver();
    }
    #endif

    /* --- TIMING END & REPORT --- */
    int print_timing = 1;
#ifdef MPI_ENABLED
    if (ARTRACFD_RUNMODE == 2 && ARTRACFD_MPI_RANK != 0) {
        print_timing = 0;
    }
#endif
    if (print_timing) {
        gettimeofday(&end_wall, NULL);
        
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);

        double real_time = (end_wall.tv_sec - start_wall.tv_sec) + 
                           (end_wall.tv_usec - start_wall.tv_usec) / 1000000.0;
        
        double user_time = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1000000.0;
        double sys_time  = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1000000.0;

        printf("\n------------------------------------------------------------\n");
        printf("real\t%.2f\n", real_time);
        printf("user\t%.2f\n", user_time);
        printf("sys \t%.2f\n", sys_time);
        printf("------------------------------------------------------------\n");
    }
    
    return 0;
}
