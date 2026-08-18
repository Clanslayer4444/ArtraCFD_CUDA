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
 * Header File Guards to Avoid Interdependence
 ****************************************************************************/
#ifndef ARTRACFD_COMPUTATIONAL_GEOMETRY_H_ /* if undefined */
#define ARTRACFD_COMPUTATIONAL_GEOMETRY_H_ /* set a unique marker */
/****************************************************************************
 * Required Header Files
 ****************************************************************************/
#include "commons.h"
/****************************************************************************
 * Data Structure Declarations
 ****************************************************************************/
/****************************************************************************
 * Public Functions Declaration
 ****************************************************************************/
/*
 * Polyhedron representation
 *
 * Function
 *      Convert polyhedron representation from STL to a mixture form
 *      of face-vertex mesh and winged-edge mesh.
 */
extern void ConvertPolyhedron(Polyhedron *);
extern void AllocatePolyhedronMemory(const int vertN, const int edgeN,
        const int faceN, Polyhedron *);
extern void AddEdge(const int v0, const int v1, const int f, Polyhedron *);
extern void QuickSortEdge(const int n, int e[RESTRICT][EVF]);
extern void BuildTriangle(const int fid, const Polyhedron *, Real v0[RESTRICT],
        Real v1[RESTRICT], Real v2[RESTRICT], Real e01[RESTRICT], Real e02[RESTRICT]);
/*
 * Compute geometry parameters
 *
 * Function
 *      Compute the geometric properties of each polyhedron, including bounding
 *      volume, area, volume, centroid, inertia tensor, normal. Note that the
 *      inertia tensor is relative to the body coordinates located at centroid
 *      and is computed by assuming that the density is a constant with value 1.
 */
extern void ComputeGeometryParameters(const int collapse, Geometry *const);
/*
 * Polyhedron transformation
 */
extern void TransformPolyhedron(const Real O[RESTRICT], const Real scale[RESTRICT],
        const Real angle[RESTRICT], const Real offset[RESTRICT], Polyhedron *);
/*
 * Point in polyhedron
 *
 * Function
 *      Solve point-in-polyhedron problem for triangulated polyhedron,
 *      also find the cloest face.
 */
extern int PointInPolyhedron(const Real p[RESTRICT], const Polyhedron *, int fid[RESTRICT]);
/*
 * Point triangle distance
 *
 * Function
 *     Returns the squared minimum distance from a point to a triangle,
 *     also finds the barycentric coordnates of the intersection point.
 */
extern Real PointTriangleDistance(const Real p[RESTRICT], const Real v0[RESTRICT],
        const Real e01[RESTRICT], const Real e02[RESTRICT], Real para[RESTRICT]);
/*
 * Point triangle intersection point
 *
 * Function
 *      Obtain the coordinates and normal of the intersection point,
 *      also return the distance.
 */
extern Real ComputeIntersection(const Real p[RESTRICT], const int fid,
        const Polyhedron *poly, Real pi[RESTRICT], Real N[RESTRICT]);
/*
 * Compute geometric data
 *
 * Function
 *      Compute the intersection point pi, mirror point pm, outward surface
 *      normal N of the point p regarding the face fid of the polyhedron.
 */
extern void ComputeGeometricData(const Real p[RESTRICT], const int fid, const Polyhedron *,
        Real pi[RESTRICT], Real pm[RESTRICT], Real N[RESTRICT]);
#endif
/* a good practice: end file with a newline */

