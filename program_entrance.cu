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
extern "C" {
#include "commons.h"
#include "cfd_commons.h"
#include "preprocess.h"
#include "postprocess.h"
#include "calculator.h"
#include "case_generator.h"
#include "linear_system.h"
}

#ifdef MPI_ENABLED
#include "mpi_interface.h"
#endif

#include <cuda_runtime.h>

/****************************************************************************
 * Static Function Declarations
 ****************************************************************************/
static void ConfigureProgram(Control *, Space *);
static void ShowPreamble(Control *);
static void ShowManual(void);
/****************************************************************************
 * Function Definitions
 ****************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

int EnterProgram(int argc, char *argv[], Control *control, Space *space)
{
    while ((1 < argc) && ('-' == argv[1][0])) { /* options present */
        if (3 > argc) { /* not enough arguments */
            ShowError("empty entry after: %s\n", argv[1]);
            exit(EXIT_FAILURE);
        }
        switch (argv[1][1]) { /* argv[1][1] is the actual option character */
            /* run mode: -m [gui], [serial], [omp], [mpi], [gpu] */
            case 'm':
                ++argv;
                --argc;
                if (0 == strcmp(argv[1], "gui")) {
                    control->runMode = 'i';
                    break;
                }
                if (0 == strcmp(argv[1], "serial")) {
                    control->runMode = 's';
                    break;
                }
                if (0 == strcmp(argv[1], "omp")) {
                    control->runMode = 'o';
                    break;
                }
                if (0 == strcmp(argv[1], "mpi")) {
                    control->runMode = 'm';
                    break;
                }
                if (0 == strcmp(argv[1], "gpu")) {
                    control->runMode = 'g';
                    break;
                }
                ShowError("bad option: %s\n", argv[1]);
                exit(EXIT_FAILURE);
                /* number of processors: -n nx*ny*nz */
            case 'n':
                ++argv;
                --argc;
                Sscanf(argv[1], 3, "%d*%d*%d", &(control->proc[X]),
                        &(control->proc[Y]), &(control->proc[Z]));
                break;
            default:
                ShowError("bad option: %s\n", argv[1]);
                exit(EXIT_FAILURE);
        }
        /* adjust argument list and count to consume an option */
        ++argv;
        --argc;
    }
    /* check information left */
    if (1 != argc) {
        ShowWarning("unidentified arguments ignored: %s...\n", argv[1]);
    }
    /* configure program according to inputted options */
    ConfigureProgram(control, space);
    return 0;
}

#ifdef __cplusplus
}
#endif

static void ConfigureProgram(Control *control, Space *space)
{
    Partition *const part = &(space->part);
    switch (control->runMode) {
        case 'i': /* gui mode */
            ShowPreamble(control);
            /* fall through */
        case 's': /* serial mode */
            part->proc[X] = 1;
            part->proc[Y] = 1;
            part->proc[Z] = 1;
            part->procN = 1;
            break;
        case 'o': /* omp mode */
            /* fall through */
        case 'm': /* mpi mode */
        {
            #ifdef MPI_ENABLED
            extern int ARTRACFD_MPI_NPROCS;
            /* Use MPI to determine processor count */
            int nprocs = ARTRACFD_MPI_NPROCS;
            if (nprocs > 1) {
                /* Distribute over Y dimension (slab decomposition) */
                part->proc[X] = 1;
                part->proc[Y] = nprocs;
                part->proc[Z] = 1;
                part->procN = nprocs;
                printf("  >> MPI subdomain decomposition: %d slabs along Y\n", nprocs);
            } else {
                part->proc[X] = 1;
                part->proc[Y] = 1;
                part->proc[Z] = 1;
                part->procN = 1;
            }
            #else
            /* Fallback if MPI not compiled in */
            part->proc[X] = control->proc[X];
            part->proc[Y] = control->proc[Y];
            part->proc[Z] = control->proc[Z];
            part->procN = control->proc[X] *
                control->proc[Y] * control->proc[Z];
            #endif
            break;
        }
        case 'g': /* gpu mode */
        {
            part->proc[X] = control->proc[X];
            part->proc[Y] = control->proc[Y];
            part->proc[Z] = control->proc[Z];
            part->procN = control->proc[X] *
                          control->proc[Y] * control->proc[Z];

            int deviceCount = 0;
            cudaError_t err = cudaGetDeviceCount(&deviceCount);
            if (err != cudaSuccess || deviceCount == 0) {
                fprintf(stderr, "No CUDA-capable GPU detected!\n");
                exit(EXIT_FAILURE);
            }

            printf("\n>> HPC Hardware Scan: Found %d CUDA GPUs on this node.\n", deviceCount);

            if (part->procN > 1) {      
                printf(">> WARNING: You requested %d partitions (-n), but the solver is currently running as a single-host process.\n", part->procN);
                printf(">> Single-Node Multi-GPU requires MPI integration or OpenMP host-threading. Defaulting to Device 0.\n");
            }

            int device = 0;  
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device);

            cudaSetDevice(device);
            printf(">> GPU mode enabled: Locked to device %d (%s)\n", device, prop.name);
            printf(">> Cores: %d SMs | Memory: %.1f GB\n", 
                   prop.multiProcessorCount, 
                   prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        }
        break;
    }
    return;
}
static void ShowPreamble(Control *control)
{
    ShowInfo("Session");
    ShowInfo("*                         ArtraCFD                         *\n");
    ShowInfo("*                     <By Huangrui Mo>                     *\n");
    ShowInfo("*    Copyright (C) Huangrui Mo <huangrui.mo@gmail.com>     *\n");
    ShowInfo("Session");
    ShowInfo("Enter 'help' for more information\n");
    ShowInfo("Session");
    String str = {'\0'}; /* store the current read line */
    while (1) {
        ShowInfo("\nArtraCFD << ");
        ParseCommand(fgets(str, sizeof str, stdin));
        ShowInfo("\n");
        if (0 == strncmp(str, "help", sizeof str)) {
            ShowInfo("Options in gui environment:\n");
            ShowInfo("[help]    show this information\n");
            ShowInfo("[init]    generate files for a sample case\n");
            ShowInfo("[solve]   solve current case in serial mode\n");
            ShowInfo("[calc]    access expression calculator\n");
            ShowInfo("[manual]  show user manual\n");
            ShowInfo("[exit]    exit program\n");
            continue;
        }
        if (0 == strncmp(str, "init", sizeof str)) {
            GenerateCaseFiles();
            ShowInfo("a sample case generated successfully\n");
            continue;
        }
        if (0 == strncmp(str, "calc", sizeof str)) {
            RunCalculator();
            continue;
        }
        if (0 == strncmp(str, "manual", sizeof str)) {
            ShowManual();
            continue;
        }
        if ('\0' == str[0]) {
            continue;
        }
        if (0 == strncmp(str, "solve", sizeof str)) {
            control->runMode = 's';
            ShowInfo("Session");
            return;
        }
        if (0 == strncmp(str, "exit", sizeof str)) {
            ShowInfo("Session");
            exit(EXIT_SUCCESS);
        }
        /* if non of above is true, then unknow commands */
        ShowWarning("unknown command: %s\n", str);
    }
}
static void ShowManual(void)
{
    ShowInfo("\n            ArtraCFD User Manual\n");
    ShowInfo("SYNOPSIS:\n");
    ShowInfo("        artracfd [-m runmode] [-n nprocessors]\n");
    ShowInfo("OPTIONS:\n");
    ShowInfo("        -m runmode        run mode: gui, serial, omp, mpi, gpu\n");
    ShowInfo("        -n nprocessors    processors per dimension: nx*ny*nz\n");
    ShowInfo("NOTES:\n");
    ShowInfo("        default run mode is gui\n");
    return;
}
/* a good practice: end file with a newline */
