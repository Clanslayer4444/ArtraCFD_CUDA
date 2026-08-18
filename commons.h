/****************************************************************************
 * ArtraCFD                                    *
 * <By Huangrui Mo>                                *
 * Copyright (C) Huangrui Mo <huangrui.mo@gmail.com>                        *
 ****************************************************************************/
#ifndef ARTRACFD_COMMONS_H_
#define ARTRACFD_COMMONS_H_

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

#ifndef RESTRICT
  #ifdef __CUDACC__
    #define RESTRICT __restrict__
  #else
    #define RESTRICT restrict
  #endif
#endif

/****************************************************************************
 * CRITICAL FIX: Safe Indexing for 2D Stability (mz=1)
 ****************************************************************************/
static inline int IndexNode(int k, int j, int i, int ny, int nx) {
    int idx = k * ny * nx + j * nx + i;
    if (idx < 0) return 0; /* Basic lower-bound safety */
    return idx;
}
/* MACRO REMOVED: Square(x) caused conflicts with weno3.c/weno5.c */

/* -------------------------------------------------------------------------- */

typedef enum {
    DIMO = 4, X = 0, Y = 1, Z = 2, DIMS = 3, PHI = 4,
    COLLAPSEN = 0, COLLAPSEX = 1, COLLAPSEY = 2, COLLAPSEZ = 3,
    COLLAPSEXY = 5, COLLAPSEXZ = 7, COLLAPSEYZ = 8, COLLAPSEXYZ = 17,
    DIMT = 3, TO = 0, TN = 1, TM = 2,
    DIMU = 5, DIMUo = 6,
    PATHN = 30, PATHSEP = 4, NONE = -1,
    WENOTHREE = 0, WENOFIVE = 1,
    OPTSPLIT = 0, OPTBYOPT = 1,
    NPART = 15, PIO = 0, PIN = 0, PWB = 1, PEB = 2, PSB = 3, PNB = 4, PFB = 5, PBB = 6,
    PWG = 7, PEG = 8, PSG = 9, PNG = 10, PFG = 11, PBG = 12, PHY = 13, PAL = 14,
    LIMIT = 2, MIN = 0, MAX = 1,
    NBC = 7, INFLOW = 0, OUTFLOW = 1, SLIPWALL = 2, NOSLIPWALL = 3, PERIODIC = 4, VARBC = 6,
    NIC = 10, ICGLOBAL = 0, ICPLANE = 1, ICSPHERE = 2, ICBOX = 3, ICCYLINDER = 4, POSIC = 7, VARIC = 5,
    DIMTK = 2, POLYN = 3, EVF = 4,
    NPROBE = 5, PROPT = 0, PROLN = 1, PROCV = 2, PROFC = 3, PROSD = 4, POSLN = 7,
    STR = 200, VARSTR = 100,
} ComConst;

typedef double Real;
typedef char String[STR];
typedef int IntVec[DIMS];
typedef Real RealVec[DIMS];

extern const Real PI;

typedef struct {
    int did; int fid; int lid; int gst;
    Real U[DIMT][DIMU];
} Node;

typedef struct {
    IntVec m; IntVec n; IntVec ng; int gl; int collapse;
    RealVec d; RealVec dd; Real tinyL;
    int ns[NPART][DIMS][LIMIT]; int np[DIMS][DIMS][LIMIT];
    int path[PATHN][DIMS]; int pathSep[PATHSEP];
    int *RESTRICT typeBC; int (*RESTRICT N)[DIMS]; Real (*RESTRICT varBC)[VARBC];
    int nIC; int *RESTRICT typeIC; Real (*RESTRICT posIC)[POSIC]; char (*RESTRICT varIC)[VARIC][VARSTR];
    Real domain[DIMS][LIMIT]; IntVec proc; int procN;
} Partition;

typedef struct {
    RealVec N; RealVec v0; RealVec v1; RealVec v2;
} Facet;

typedef struct {
    int gid; IntVec N;
} Collision;

typedef struct {
    int faceN; int edgeN; int vertN; int state; int mid;
    Real r; RealVec O; Real I[DIMS][DIMS];
    Real V[DIMTK][DIMS]; Real W[DIMTK][DIMS]; Real at[DIMTK][DIMS];
    RealVec g; Real ar[DIMTK][DIMS]; RealVec Fp; RealVec Fv; RealVec Tt;
    Real to; Real rho; Real T; Real cf; Real area; Real volume;
    Real box[DIMS][LIMIT];
    int (*RESTRICT f)[POLYN]; Real (*RESTRICT Nf)[DIMS];
    int (*RESTRICT e)[EVF]; Real (*RESTRICT Ne)[DIMS];
    Real (*RESTRICT v)[DIMS]; Real (*RESTRICT Nv)[DIMS];
    Facet *facet;
} Polyhedron;

typedef struct {
    int totN; int sphN; int stlN; int colN;
    Polyhedron *poly; Collision *col;
} Geometry;

typedef struct { Real eos; } Material;

typedef struct {
    Node *node; Geometry geo; Partition part;
} Space;

typedef struct {
    int restart; int stepN; int stepC;
    int dataN[NPROBE]; int dataW[NPROBE]; int dataStreamer; int dataC;
    Real end; Real now; Real numCFL;
    Real (*RESTRICT pp)[DIMS]; Real (*RESTRICT lp)[POSLN];
} Time;

typedef struct {
    int tScheme; int sScheme; int sL; int sR;
    int multidim; int jacobMean; int fluxSplit; int psi; int ibmLayer;
    int mid; int gState; int sState;
    Real refMa; Real refMu; Real gamma; Real gasR; Real cv;
    Real refL; Real refRho; Real refV; Real refT;
    RealVec g; Material *mat;
} Model;

typedef struct {
    char runMode; IntVec proc;
} Control;

extern int ParseCommand(char *cmdstr);
extern char *ParseFormat(char *fmt);
extern void ShowError(const char *fmt, ...);
extern void ShowWarning(const char *fmt, ...);
extern void verror(const char *prefix, const char *fmt, va_list args);
extern void ShowInfo(const char *fmt, ...);
extern void *AssignStorage(size_t size);
extern void RetrieveStorage(void *pointer);
extern void ReadInLine(FILE *fp, const char *line);
extern void WriteToLine(FILE *fp, const char *line);
extern FILE *Fopen(const char *fname, const char *mode);
extern void Fread(void *ptr, size_t size, size_t n, FILE *stream);
extern void Fscanf(FILE *stream, const int n, const char *fmt, ...);
extern void Sscanf(const char *str, const int n, const char *fmt, ...);
extern void Sread(FILE *stream, const int n, const char *fmt, ...);

#endif