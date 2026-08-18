#include "immersed_boundary.h"
#include <stdio.h> 
#include <math.h> 
#include <stdlib.h> 
#include <float.h> 
#include <string.h> 
#include "computational_geometry.h"
#include "cfd_commons.h"
#include "commons.h"
#include "gpu_ibm.cuh"

typedef enum { R = 2, INTERL = 0, INTERG = 1, TYPED = -1, TYPEF = -2, TYPEL = -3 } IbmConst;
static void InitializeGeometricField(Space *);
static void SetDomainField(Space *);
static void SetInterfacialField(Space *, const Model *);
static int GetInterState(const int, const int, const int, const int, const int, const int, const int [RESTRICT][DIMS], const Node *const, const Partition *const);
static void ApplyWeighting(const Real [RESTRICT], const Real, Real, Real [RESTRICT], Real [RESTRICT]);
static Real InverseDistanceWeighting(const int, const int [RESTRICT], const Real [RESTRICT], const int, const int, const int, const Partition *const, const Node *const, const Model *, Real [RESTRICT]);
static void ReconstructFlow(const int, const int [RESTRICT], const Real [RESTRICT], const int, const int, const int, const Polyhedron *, const Partition *const, const Node *const, const Model *, const Real [RESTRICT], const Real [RESTRICT], Real [RESTRICT], Real [RESTRICT]);

void ComputeGeometricField(Space *space, const Model *model) { InitializeGeometricField(space); SetDomainField(space); SetInterfacialField(space, model); }

static void InitializeGeometricField(Space *space) {
    const Partition *const part = &(space->part); Node *const node = space->node; const Geometry *const geo = &(space->geo); const Polyhedron *poly = NULL; int idx = 0; int gid = 0; 
    int total_nY = part->ns[PAL][Y][MAX] - part->ns[PAL][Y][MIN];
    int total_nX = part->ns[PAL][X][MAX] - part->ns[PAL][X][MIN];
    for (int k = part->ns[PIN][Z][MIN]; k < part->ns[PIN][Z][MAX]; ++k) { for (int j = part->ns[PIN][Y][MIN]; j < part->ns[PIN][Y][MAX]; ++j) { for (int i = part->ns[PIN][X][MIN]; i < part->ns[PIN][X][MAX]; ++i) { 
        idx = IndexNode(k, j, i, total_nY, total_nX); 
        gid = node[idx].did; node[idx].gst = node[idx].did; if (0 >= gid) { node[idx].fid = 0; continue; } poly = geo->poly + gid - 1; if (1 == poly->state) { continue; } if (0 < node[idx].lid) { node[idx].did = 0; } 
    } } }
}

static void SetDomainField(Space *space) {
    const Partition *const part = &(space->part); Node *const node = space->node; const Geometry *const geo = &(space->geo); const IntVec nMin = {part->ns[PIN][X][MIN], part->ns[PIN][Y][MIN], part->ns[PIN][Z][MIN]}; const IntVec nMax = {part->ns[PIN][X][MAX], part->ns[PIN][Y][MAX], part->ns[PIN][Z][MAX]}; const RealVec sMin = {part->domain[X][MIN], part->domain[Y][MIN], part->domain[Z][MIN]}; const RealVec d = {part->d[X], part->d[Y], part->d[Z]}; const RealVec dd = {part->dd[X], part->dd[Y], part->dd[Z]}; const IntVec ng = {part->ng[X], part->ng[Y], part->ng[Z]}; const Polyhedron *poly = NULL; int box[DIMS][LIMIT] = {{0}}; int fid = 0; int idx = 0; RealVec p = {0.0}; 
    int total_nY = part->ns[PAL][Y][MAX] - part->ns[PAL][Y][MIN];
    int total_nX = part->ns[PAL][X][MAX] - part->ns[PAL][X][MIN];
    for (int n = 0; n < geo->totN; ++n) { poly = geo->poly + n; if (1 == poly->state) { continue; } for (int s = 0; s < DIMS; ++s) { box[s][MIN] = ConfineSpace(MapNode(poly->box[s][MIN], sMin[s], dd[s], ng[s]), nMin[s], nMax[s]); box[s][MAX] = ConfineSpace(MapNode(poly->box[s][MAX], sMin[s], dd[s], ng[s]), nMin[s], nMax[s]) + 1; } for (int k = box[Z][MIN]; k < box[Z][MAX]; ++k) { for (int j = box[Y][MIN]; j < box[Y][MAX]; ++j) { for (int i = box[X][MIN]; i < box[X][MAX]; ++i) { 
        idx = IndexNode(k, j, i, total_nY, total_nX); 
        if (0 != node[idx].did) { continue; } p[X] = MapPoint(i, sMin[X], d[X], ng[X]); p[Y] = MapPoint(j, sMin[Y], d[Y], ng[Y]); p[Z] = MapPoint(k, sMin[Z], d[Z], ng[Z]); if (0 >= poly->faceN) { if (poly->r * poly->r >= Dist2(poly->O, p)) { node[idx].did = n + 1; node[idx].fid = 0; } } else { if (PointInPolyhedron(p, poly, &fid)) { node[idx].did = n + 1; node[idx].fid = fid; } } 
    } } } }
}

static void SetInterfacialField(Space *space, const Model *model) {
    const Partition *const part = &(space->part); Node *const node = space->node; int idx = 0; const int sd = 0; IntVec n = {0}; RealVec p = {0.0}; Real Uo[DIMUo] = {0.0}; Real weightSum = 0.0;
    int total_nY = part->ns[PAL][Y][MAX] - part->ns[PAL][Y][MIN];
    int total_nX = part->ns[PAL][X][MAX] - part->ns[PAL][X][MIN];
    for (int k = part->ns[PIN][Z][MIN]; k < part->ns[PIN][Z][MAX]; ++k) { for (int j = part->ns[PIN][Y][MIN]; j < part->ns[PIN][Y][MAX]; ++j) { for (int i = part->ns[PIN][X][MIN]; i < part->ns[PIN][X][MAX]; ++i) { 
        idx = IndexNode(k, j, i, total_nY, total_nX); 
        if ((node[idx].gst != node[idx].did) && (sd == node[idx].did)) { n[X] = i; n[Y] = j; n[Z] = k; p[X] = MapPoint(i, part->domain[X][MIN], part->d[X], part->ng[X]); p[Y] = MapPoint(j, part->domain[Y][MIN], part->d[Y], part->ng[Y]); p[Z] = MapPoint(k, part->domain[Z][MIN], part->d[Z], part->ng[Z]); weightSum = InverseDistanceWeighting(TO, n, p, R, TYPEF, node[idx].did, part, node, model, Uo); Normalize(DIMUo, weightSum, Uo); Uo[0] = Uo[4] / (Uo[5] * model->gasR); MapConservative(model->gamma, Uo, node[idx].U[TO]); node[idx].fid = NONE; } node[idx].lid = 0; node[idx].gst = 0; if (sd == node[idx].did) { continue; } node[idx].lid = GetInterState(INTERL, k, j, i, node[idx].did, part->pathSep[0], part->path, node, part); if ((0 < node[idx].lid) && (sd != node[idx].did)) { node[idx].gst = GetInterState(INTERG, k, j, i, sd, part->pathSep[0], part->path, node, part); } 
    } } }
}

static int GetInterState(const int sid, const int k, const int j, const int i, const int did, const int end, const int path[RESTRICT][DIMS], const Node *const node, const Partition *const part) {
    int idx = 0, ih = 0, jh = 0, kh = 0, flag = 0;
    int is2D = (part->n[Z] < 2);
    int total_nY = part->ns[PAL][Y][MAX] - part->ns[PAL][Y][MIN];
    int total_nX = part->ns[PAL][X][MAX] - part->ns[PAL][X][MIN];
    for (int n = 0; n < end; ++n) {
        if (is2D && path[n][Z] != 0) continue;
        kh = k + path[n][Z]; jh = j + path[n][Y]; ih = i + path[n][X];
        if (!InPartBox(kh, jh, ih, part->ns[PIN])) continue;
        idx = IndexNode(kh, jh, ih, total_nY, total_nX);
        switch (sid) { case INTERL: if (did != node[idx].did) flag = 1; break; case INTERG: if (did == node[idx].did) flag = 1; break; default: break; }
        if (1 == flag) { for (int r = 1; r <= part->gl; ++r) { if (part->pathSep[r] > n) return r; } }
    }
    return 0;
}

void TreatImmersedBoundary(const int tn, Space *space, const Model *model) {
    const Partition *const part = &(space->part);
    Node *const node = space->node;
    const Geometry *const geo = &(space->geo);
    const IntVec nMin = {part->ns[PIN][X][MIN], part->ns[PIN][Y][MIN], part->ns[PIN][Z][MIN]};
    const IntVec nMax = {part->ns[PIN][X][MAX], part->ns[PIN][Y][MAX], part->ns[PIN][Z][MAX]};
    const RealVec sMin = {part->domain[X][MIN], part->domain[Y][MIN], part->domain[Z][MIN]};
    const RealVec d = {part->d[X], part->d[Y], part->d[Z]};
    const RealVec dd = {part->dd[X], part->dd[Y], part->dd[Z]};
    const IntVec ng = {part->ng[X], part->ng[Y], part->ng[Z]};
    const Polyhedron *poly = NULL;
    
    int idx = 0;
    IntVec nI = {0}, nG = {0};
    RealVec pG = {0.0}, pO = {0.0}, pI = {0.0}, N = {0.0};
    Real UoG[DIMUo] = {0.0}, UoO[DIMUo] = {0.0}, UoI[DIMUo] = {0.0};
    Real weightSum = 0.0;
    int box[DIMS][LIMIT] = {{0}};
    
    int total_nY = part->ns[PAL][Y][MAX] - part->ns[PAL][Y][MIN];
    int total_nX = part->ns[PAL][X][MAX] - part->ns[PAL][X][MIN];

    for (int n = 0; n < geo->totN; ++n) {
        poly = geo->poly + n;
        
        for (int s = 0; s < DIMS; ++s) {
            box[s][MIN] = ConfineSpace(MapNode(poly->box[s][MIN], sMin[s], dd[s], ng[s]), nMin[s], nMax[s]);
            box[s][MAX] = ConfineSpace(MapNode(poly->box[s][MAX], sMin[s], dd[s], ng[s]), nMin[s], nMax[s]) + 1;
        }
        
        for (int r = 1; r <= part->gl; ++r) {
            for (int k = box[Z][MIN]; k < box[Z][MAX]; ++k) {
                for (int j = box[Y][MIN]; j < box[Y][MAX]; ++j) {
                    for (int i = box[X][MIN]; i < box[X][MAX]; ++i) {
                        
                        idx = IndexNode(k, j, i, total_nY, total_nX);
                        
                        if ((r != node[idx].gst) || (n + 1 != node[idx].did)) {
                            continue;
                        }
                        
                        pG[X] = MapPoint(i, sMin[X], d[X], ng[X]);
                        pG[Y] = MapPoint(j, sMin[Y], d[Y], ng[Y]);
                        pG[Z] = MapPoint(k, sMin[Z], d[Z], ng[Z]);
                        
                        if (model->ibmLayer >= r) {
                            ComputeGeometricData(pG, node[idx].fid, poly, pO, pI, N);
                            nI[X] = MapNode(pI[X], sMin[X], dd[X], ng[X]);
                            nI[Y] = MapNode(pI[Y], sMin[Y], dd[Y], ng[Y]);
                            nI[Z] = MapNode(pI[Z], sMin[Z], dd[Z], ng[Z]);
                            ReconstructFlow(tn, nI, pI, R, TYPED, 0, poly, part, node, model, pO, N, UoO, UoI);
                            DoMethodOfImage(UoI, UoO, UoG);
                        } else {
                            nG[X] = i; nG[Y] = j; nG[Z] = k;
                            weightSum = InverseDistanceWeighting(tn, nG, pG, 1, r - 1, n + 1, part, node, model, UoG);
                            Normalize(DIMUo, weightSum, UoG);
                        }
                        
                        UoG[0] = UoG[4] / (UoG[5] * model->gasR);
                        MapConservative(model->gamma, UoG, node[idx].U[tn]);
                        
                    }
                }
            }
        }
    }
}

void DoMethodOfImage(const Real UoI[RESTRICT], const Real UoO[RESTRICT], Real UoG[RESTRICT]) {
    UoG[1] = UoO[1] + UoO[1] - UoI[1]; UoG[2] = UoO[2] + UoO[2] - UoI[2]; UoG[3] = UoO[3] + UoO[3] - UoI[3]; UoG[4] = UoI[4]; UoG[5] = UoI[5];
}

static void ReconstructFlow(const int tn, const int n[RESTRICT], const Real p[RESTRICT], const int h, const int type, const int did, const Polyhedron *poly, const Partition *const part, const Node *const node, const Model *model, const Real pO[RESTRICT], const Real N[RESTRICT], Real UoO[RESTRICT], Real Uo[RESTRICT]) {
    const Real zero = 0.0; const Real one = 1.0; Real weightSum = InverseDistanceWeighting(tn, n, p, h, type, did, part, node, model, Uo); const Real weight = one / weightSum; RealVec Vs = {zero}; const RealVec r = {pO[X] - poly->O[X], pO[Y] - poly->O[Y], pO[Z] - poly->O[Z]}; Cross(poly->W[TO], r, Vs); Vs[X] = poly->V[TO][X] + Vs[X]; Vs[Y] = poly->V[TO][Y] + Vs[Y]; Vs[Z] = poly->V[TO][Z] + Vs[Z]; if (zero < poly->cf) { UoO[1] = Vs[X]; UoO[2] = Vs[Y]; UoO[3] = Vs[Z]; } else { const RealVec V = {Uo[1] * weight, Uo[2] * weight, Uo[3] * weight}; RealVec Ta = {zero}; RealVec Tb = {zero}; Real RHS[DIMS] = {zero}; OrthogonalSpace(N, Ta, Tb); RHS[X] = Dot(Vs, N); RHS[Y] = Dot(V, Ta); RHS[Z] = Dot(V, Tb); UoO[1] = N[X] * RHS[X] + Ta[X] * RHS[Y] + Tb[X] * RHS[Z]; UoO[2] = N[Y] * RHS[X] + Ta[Y] * RHS[Y] + Tb[Y] * RHS[Z]; UoO[3] = N[Z] * RHS[X] + Ta[Z] * RHS[Y] + Tb[Z] * RHS[Z]; } UoO[4] = Uo[4] * weight; if (poly->T <= zero) { UoO[5] = Uo[5] * weight; } else { UoO[5] = poly->T; } ApplyWeighting(UoO, part->tinyL, Dist2(p, pO), &weightSum, Uo); Normalize(DIMUo, weightSum, Uo);
}

static Real InverseDistanceWeighting(const int tn, const int n[RESTRICT], const Real p[RESTRICT], const int h, const int type, const int did, const Partition *const part, const Node *const node, const Model *model, Real Uo[RESTRICT]) {
    int idx = 0; const RealVec sMin = {part->domain[X][MIN], part->domain[Y][MIN], part->domain[Z][MIN]}; const RealVec d = {part->d[X], part->d[Y], part->d[Z]}; const IntVec ng = {part->ng[X], part->ng[Y], part->ng[Z]}; Real Uoh[DIMUo] = {0.0}; RealVec ph = {0.0}; IntVec nh = {0}; Real weightSum = 0.0; memset(Uo, 0, DIMUo * sizeof(*Uo));
    int is2D = (part->n[Z] < 2);
    int total_nY = part->ns[PAL][Y][MAX] - part->ns[PAL][Y][MIN];
    int total_nX = part->ns[PAL][X][MAX] - part->ns[PAL][X][MIN];
    for (int r = h, tally = 0; 0 == tally; ++r) {
        for (int kh = -r; kh <= r; ++kh) {
            if (is2D && kh != 0) continue;
            for (int jh = -r; jh <= r; ++jh) { for (int ih = -r; ih <= r; ++ih) { nh[X] = n[X] + ih; nh[Y] = n[Y] + jh; nh[Z] = n[Z] + kh; if (!InPartBox(nh[Z], nh[Y], nh[X], part->ns[PIN])) continue; 
                idx = IndexNode(nh[Z], nh[Y], nh[X], total_nY, total_nX); 
                if (did != node[idx].did) continue; switch (type) { case TYPED: break; case TYPEF: if ((did != node[idx].gst) || (0 > node[idx].fid)) continue; break; default: if (type != node[idx].gst) continue; break; } ++tally; ph[X] = MapPoint(nh[X], sMin[X], d[X], ng[X]); ph[Y] = MapPoint(nh[Y], sMin[Y], d[Y], ng[Y]); ph[Z] = MapPoint(nh[Z], sMin[Z], d[Z], ng[Z]); MapPrimitive(model->gamma, model->gasR, node[idx].U[tn], Uoh); ApplyWeighting(Uoh, part->tinyL, Dist2(p, ph), &weightSum, Uo); 
            } }
        }
    }
    return weightSum;
}

static void ApplyWeighting(const Real Uoh[RESTRICT], const Real tiny, Real weight, Real weightSum[RESTRICT], Real Uo[RESTRICT]) {
    const Real one = 1.0; if (tiny > weight) weight = tiny; weight = one / weight; for (int n = 0; n < DIMUo; ++n) Uo[n] = Uo[n] + Uoh[n] * weight; *weightSum = *weightSum + weight;
}

IBM_Stencil *h_ibm_map = NULL;
int num_ibm_stencils = 0;

static void ExtractIDWStencil(const IntVec n, const RealVec p, const int h, const int type, const int did, const Partition *const part, const Node *const node, IBM_Stencil *stencil) {
    int idx = 0; 
    const RealVec sMin = {part->domain[X][MIN], part->domain[Y][MIN], part->domain[Z][MIN]}; 
    const RealVec d = {part->d[X], part->d[Y], part->d[Z]}; 
    const IntVec ng = {part->ng[X], part->ng[Y], part->ng[Z]}; 
    RealVec ph = {0.0}; 
    IntVec nh = {0}; 
    
    stencil->num_neighbors = 0;
    stencil->weight_sum = 0.0;
    
    int is2D = (part->n[Z] < 2);
    int total_nY = part->ns[PAL][Y][MAX] - part->ns[PAL][Y][MIN];
    int total_nX = part->ns[PAL][X][MAX] - part->ns[PAL][X][MIN];
    
    for (int r = h, tally = 0; 0 == tally; ++r) {
        for (int kh = -r; kh <= r; ++kh) {
            if (is2D && kh != 0) continue;
            for (int jh = -r; jh <= r; ++jh) { 
                for (int ih = -r; ih <= r; ++ih) { 
                    nh[X] = n[X] + ih; nh[Y] = n[Y] + jh; nh[Z] = n[Z] + kh; 
                    if (!InPartBox(nh[Z], nh[Y], nh[X], part->ns[PIN])) continue;
                    
                    idx = IndexNode(nh[Z], nh[Y], nh[X], total_nY, total_nX);
                    if (did != node[idx].did) continue; 
                    
                    switch (type) { 
                        case TYPED: break; 
                        case TYPEF: if ((did != node[idx].gst) || (0 > node[idx].fid)) continue; break; 
                        default: if (type != node[idx].gst) continue; break; 
                    } 
                    
                    ++tally; 
                    ph[X] = MapPoint(nh[X], sMin[X], d[X], ng[X]); 
                    ph[Y] = MapPoint(nh[Y], sMin[Y], d[Y], ng[Y]); 
                    ph[Z] = MapPoint(nh[Z], sMin[Z], d[Z], ng[Z]); 
                    
                    Real dist2 = Dist2(p, ph);
                    if (part->tinyL > dist2) dist2 = part->tinyL;
                    Real weight = 1.0 / dist2;
                    
                    if (stencil->num_neighbors < MAX_STENCIL) {
                        stencil->neighbor_indices[stencil->num_neighbors] = idx;
                        stencil->weights[stencil->num_neighbors] = weight;
                        stencil->weight_sum += weight;
                        stencil->num_neighbors++;
                    }
                } 
            }
        }
    }
}

void BuildIBMMap(Space *space, const Model *model) {
    const Partition *const part = &(space->part); 
    Node *const node = space->node; 
    const Geometry *const geo = &(space->geo); 
    const IntVec nMin = {part->ns[PIN][X][MIN], part->ns[PIN][Y][MIN], part->ns[PIN][Z][MIN]}; 
    const IntVec nMax = {part->ns[PIN][X][MAX], part->ns[PIN][Y][MAX], part->ns[PIN][Z][MAX]}; 
    const RealVec sMin = {part->domain[X][MIN], part->domain[Y][MIN], part->domain[Z][MIN]}; 
    const RealVec d = {part->d[X], part->d[Y], part->d[Z]}; 
    const RealVec dd = {part->dd[X], part->dd[Y], part->dd[Z]}; 
    const IntVec ng = {part->ng[X], part->ng[Y], part->ng[Z]}; 
    const Polyhedron *poly = NULL; 
    int box[DIMS][LIMIT] = {{0}}; 
    int idx = 0;
    
    int total_nY = part->ns[PAL][Y][MAX] - part->ns[PAL][Y][MIN];
    int total_nX = part->ns[PAL][X][MAX] - part->ns[PAL][X][MIN];

    /* 1. Count Ghost Cells to allocate exact VRAM */
    num_ibm_stencils = 0;
    for (int n = 0; n < geo->totN; ++n) { 
        poly = geo->poly + n; 
        for (int s = 0; s < DIMS; ++s) { 
            box[s][MIN] = ConfineSpace(MapNode(poly->box[s][MIN], sMin[s], dd[s], ng[s]), nMin[s], nMax[s]); 
            box[s][MAX] = ConfineSpace(MapNode(poly->box[s][MAX], sMin[s], dd[s], ng[s]), nMin[s], nMax[s]) + 1; 
        } 
        for (int r = 1; r <= part->gl; ++r) { 
            for (int k = box[Z][MIN]; k < box[Z][MAX]; ++k) { 
                for (int j = box[Y][MIN]; j < box[Y][MAX]; ++j) { 
                    for (int i = box[X][MIN]; i < box[X][MAX]; ++i) {
                        idx = IndexNode(k, j, i, total_nY, total_nX);
                        if ((r != node[idx].gst) || (n + 1 != node[idx].did)) continue;
                        num_ibm_stencils++;
                    }
                }
            }
        }
    }

    if (num_ibm_stencils == 0) return;
    h_ibm_map = (IBM_Stencil*)malloc(num_ibm_stencils * sizeof(IBM_Stencil));

    /* 2. Populate Map */
    int stencil_idx = 0;
    for (int n = 0; n < geo->totN; ++n) { 
        poly = geo->poly + n; 
        for (int s = 0; s < DIMS; ++s) { 
            box[s][MIN] = ConfineSpace(MapNode(poly->box[s][MIN], sMin[s], dd[s], ng[s]), nMin[s], nMax[s]); 
            box[s][MAX] = ConfineSpace(MapNode(poly->box[s][MAX], sMin[s], dd[s], ng[s]), nMin[s], nMax[s]) + 1; 
        } 
        for (int r = 1; r <= part->gl; ++r) { 
            for (int k = box[Z][MIN]; k < box[Z][MAX]; ++k) { 
                for (int j = box[Y][MIN]; j < box[Y][MAX]; ++j) { 
                    for (int i = box[X][MIN]; i < box[X][MAX]; ++i) {
                        idx = IndexNode(k, j, i, total_nY, total_nX);
                        if ((r != node[idx].gst) || (n + 1 != node[idx].did)) continue;
                        
                        RealVec pG = { MapPoint(i, sMin[X], d[X], ng[X]), MapPoint(j, sMin[Y], d[Y], ng[Y]), MapPoint(k, sMin[Z], d[Z], ng[Z]) };
                        
                        h_ibm_map[stencil_idx].ghost_idx = idx;
                        h_ibm_map[stencil_idx].is_mirror = (model->ibmLayer >= r) ? 1 : 0;
                        
                        IntVec n_search = {i, j, k};
                        RealVec p_search = {pG[X], pG[Y], pG[Z]};
                        int h_search = 1;
                        int type_search = r - 1;

                        if (h_ibm_map[stencil_idx].is_mirror) {
                            RealVec pO, pI, N;
                            ComputeGeometricData(pG, node[idx].fid, poly, pO, pI, N);
                            
                            n_search[X] = MapNode(pI[X], sMin[X], dd[X], ng[X]);
                            n_search[Y] = MapNode(pI[Y], sMin[Y], dd[Y], ng[Y]);
                            n_search[Z] = MapNode(pI[Z], sMin[Z], dd[Z], ng[Z]);
                            p_search[X] = pI[X]; p_search[Y] = pI[Y]; p_search[Z] = pI[Z];
                            h_search = R;
                            type_search = TYPED;
                            
                            h_ibm_map[stencil_idx].N[0] = N[X]; h_ibm_map[stencil_idx].N[1] = N[Y]; h_ibm_map[stencil_idx].N[2] = N[Z];
                            h_ibm_map[stencil_idx].pO[0] = pO[X]; h_ibm_map[stencil_idx].pO[1] = pO[Y]; h_ibm_map[stencil_idx].pO[2] = pO[Z];
                            h_ibm_map[stencil_idx].poly_T = poly->T; h_ibm_map[stencil_idx].poly_cf = poly->cf;
                            h_ibm_map[stencil_idx].poly_V[0] = poly->V[TO][X]; h_ibm_map[stencil_idx].poly_V[1] = poly->V[TO][Y]; h_ibm_map[stencil_idx].poly_V[2] = poly->V[TO][Z];
                            h_ibm_map[stencil_idx].poly_W[0] = poly->W[TO][X]; h_ibm_map[stencil_idx].poly_W[1] = poly->W[TO][Y]; h_ibm_map[stencil_idx].poly_W[2] = poly->W[TO][Z];
                            h_ibm_map[stencil_idx].poly_O[0] = poly->O[X]; h_ibm_map[stencil_idx].poly_O[1] = poly->O[Y]; h_ibm_map[stencil_idx].poly_O[2] = poly->O[Z];
                            
                            /* Adjust distance weight for final method of image scaling */
                            h_ibm_map[stencil_idx].weight_sum += (1.0 / part->tinyL); 
                        }

                        int did_search = h_ibm_map[stencil_idx].is_mirror ? 0 : (n + 1);
                        ExtractIDWStencil(n_search, p_search, h_search, type_search, did_search, part, node, &h_ibm_map[stencil_idx]);
                        
                        stencil_idx++;
                    }
                }
            }
        }
    }
    printf("\n[CPU DIAGNOSTIC] sizeof(IBM_Stencil) = %zu bytes\n", sizeof(IBM_Stencil));
    
}
