#include "boundary_treatment.h"
#include <stdio.h>
#include "immersed_boundary.h"
#include "cfd_commons.h"
#include "commons.h"

static void ApplyBoundaryCondition(const int, const int, int [RESTRICT][LIMIT], const int, Space *, const Model *);
static void EnforceZeroGradient(const Real [RESTRICT], Real [RESTRICT]);

void TreatBoundary(const int tn, Space *space, const Model *model) {
    const Partition *const part = &(space->part);
    int is2D = (part->n[Z] < 2);
    
    /* 1. IBM CHECK: If you want IBM in 2D, you must provide immersed_boundary.c to be fixed.
       For now, we skip it in 2D to stop the crash. */
    TreatImmersedBoundary(tn, space, model);

    const IntVec ng = {part->ng[X], part->ng[Y], part->ng[Z]};
    const int R = MaxInt(ng[X], MaxInt(ng[Y], ng[Z]));
    int box[DIMS][LIMIT] = {{0}}; 
    
    for (int r = 0; r <= R; ++r) { 
        for (int p = PWB; p <= PBB; ++p) {
            
            if (is2D && (p == PFB || p == PBB)) continue;

            const IntVec N = {part->N[p][X], part->N[p][Y], part->N[p][Z]};
            for (int s = 0; s < DIMS; ++s) { 
                box[s][MIN] = part->ns[p][s][MIN] + MinInt(r, ng[s]) * (N[s] - !N[s]);
                box[s][MAX] = part->ns[p][s][MAX] + MinInt(r, ng[s]) * (N[s] + !N[s]) - (ng[s] < r) * (!!N[s]);
            }
            if ((box[X][MIN] >= box[X][MAX]) || (box[Y][MIN] >= box[Y][MAX]) || (box[Z][MIN] >= box[Z][MAX])) continue;
            ApplyBoundaryCondition(p, r, box, tn, space, model);
        }
    }
}
static void ApplyBoundaryCondition(const int p, const int r, int box[RESTRICT][LIMIT], const int tn, Space *space, const Model *model) {
    const Partition *const part = &(space->part);
    Node *const node = space->node;
    const Real zero = 0.0;
    const Real UoGiven[DIMUo] = {part->varBC[p][0], part->varBC[p][1], part->varBC[p][2], part->varBC[p][3], part->varBC[p][4], part->varBC[p][5]};
    const IntVec N = {part->N[p][X], part->N[p][Y], part->N[p][Z]};
    const IntVec LN = {part->m[X] * N[X], part->m[Y] * N[Y], part->m[Z] * N[Z]};
    Real *RESTRICT UG = NULL; Real *RESTRICT UI = NULL; Real *RESTRICT UO = NULL; Real *RESTRICT Uh = NULL;
    int idxG = 0, idxI = 0, idxO = 0, idxh = 0;
    Real UoG[DIMUo] = {zero}, UoI[DIMUo] = {zero}, UoO[DIMUo] = {zero}, Uoh[DIMUo] = {zero};
    
    
    int total_nY = part->ns[PAL][Y][MAX] - part->ns[PAL][Y][MIN];
    int total_nX = part->ns[PAL][X][MAX] - part->ns[PAL][X][MIN];
    /* ----------------------- */

    for (int k = box[Z][MIN]; k < box[Z][MAX]; ++k) {
        for (int j = box[Y][MIN]; j < box[Y][MAX]; ++j) {
            for (int i = box[X][MIN]; i < box[X][MAX]; ++i) {
                if (0 != r) { 
                    idxG = IndexNode(k, j, i, total_nY, total_nX);
                    UG = node[idxG].U[tn];
                    switch (part->typeBC[p]) {
                        case SLIPWALL: case NOSLIPWALL:
                            idxO = IndexNode(k - r*N[Z], j - r*N[Y], i - r*N[X], total_nY, total_nX);
                            UO = node[idxO].U[tn]; MapPrimitive(model->gamma, model->gasR, UO, UoO);
                            idxI = IndexNode(k - 2*r*N[Z], j - 2*r*N[Y], i - 2*r*N[X], total_nY, total_nX);
                            UI = node[idxI].U[tn]; MapPrimitive(model->gamma, model->gasR, UI, UoI);
                            DoMethodOfImage(UoI, UoO, UoG);
                            UoG[0] = UoG[4] / (UoG[5] * model->gasR); MapConservative(model->gamma, UoG, UG);
                            break;
                        case PERIODIC:
                            idxh = IndexNode(k - LN[Z], j - LN[Y], i - LN[X], total_nY, total_nX);
                            Uh = node[idxh].U[tn]; EnforceZeroGradient(Uh, UG);
                            break;
                        default:
                            idxh = IndexNode(k - N[Z], j - N[Y], i - N[X], total_nY, total_nX);
                            Uh = node[idxh].U[tn]; EnforceZeroGradient(Uh, UG);
                            break;
                    }
                    continue;
                }
                idxO = IndexNode(k, j, i, total_nY, total_nX);
                UO = node[idxO].U[tn];
                switch (part->typeBC[p]) { 
                    case INFLOW: MapConservative(model->gamma, UoGiven, UO); break;
                    case OUTFLOW:
                        idxh = IndexNode(k - N[Z], j - N[Y], i - N[X], total_nY, total_nX);
                        Uh = node[idxh].U[tn]; EnforceZeroGradient(Uh, UO); break;
                    case SLIPWALL: 
                        idxh = IndexNode(k - N[Z], j - N[Y], i - N[X], total_nY, total_nX);
                        Uh = node[idxh].U[tn]; MapPrimitive(model->gamma, model->gasR, Uh, Uoh);
                        UoO[1] = (!N[X]) * Uoh[1]; UoO[2] = (!N[Y]) * Uoh[2]; UoO[3] = (!N[Z]) * Uoh[3];
                        UoO[4] = Uoh[4]; UoO[5] = (zero >= UoGiven[5]) ? Uoh[5] : UoGiven[5];
                        UoO[0] = UoO[4] / (UoO[5] * model->gasR); MapConservative(model->gamma, UoO, UO); break;
                    case NOSLIPWALL:
                        idxh = IndexNode(k - N[Z], j - N[Y], i - N[X], total_nY, total_nX);
                        Uh = node[idxh].U[tn]; MapPrimitive(model->gamma, model->gasR, Uh, Uoh);
                        UoO[1] = zero; UoO[2] = zero; UoO[3] = zero;
                        UoO[4] = Uoh[4]; UoO[5] = (zero >= UoGiven[5]) ? Uoh[5] : UoGiven[5];
                        UoO[0] = UoO[4] / (UoO[5] * model->gasR); MapConservative(model->gamma, UoO, UO); break;
                    case PERIODIC: break;
                    default: break;
                }
            }
        }
    }
}
static void EnforceZeroGradient(const Real Uh[RESTRICT], Real U[RESTRICT]) {
    for (int n = 0; n < DIMU; ++n) U[n] = Uh[n];
}