#ifndef ARTRACFD_CFD_COMMONS_H_
#define ARTRACFD_CFD_COMMONS_H_

#include "commons.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void SymmetricAverage(const int averager, const Real gamma, const Real *RESTRICT UL, const Real *RESTRICT UR, Real *RESTRICT Uo);
extern void Eigenvalue(const int s, const Real *RESTRICT Uo, Real *RESTRICT Lambda);
extern void EigenvalueSplitting(const int splitter, const Real *RESTRICT Lambda, Real *RESTRICT LambdaP, Real *RESTRICT LambdaN);
extern void EigenvectorL(const int s, const Real gamma, const Real *RESTRICT Uo, Real (*RESTRICT L)[DIMU]);
extern void EigenvectorR(const int s, const Real *RESTRICT Uo, Real (*RESTRICT R)[DIMU]);
extern void ConvectiveFlux(const int s, const Real gamma, const Real *RESTRICT U, Real *RESTRICT F);
extern Real Viscosity(const Real T);
extern Real PrandtlNumber(void);
extern void MapPrimitive(const Real gamma, const Real gasR, const Real *RESTRICT U, Real *RESTRICT Uo);
extern Real ComputePressure(const Real gamma, const Real *RESTRICT U);
extern Real ComputeTemperature(const Real cv, const Real *RESTRICT U);
extern void MapConservative(const Real gamma, const Real *RESTRICT Uo, Real *RESTRICT U);

/* IndexNode removed (inline in commons.h) */

extern int InPartBox(const int k, const int j, const int i, const int (*RESTRICT pbox)[LIMIT]);
extern int MapNode(const Real s, const Real sMin, const Real dds, const int ng);
extern int ConfineSpace(const int n, const int nMin, const int nMax);
extern Real MapPoint(const int n, const Real sMin, const Real ds, const int ng);

extern Real MinReal(const Real x, const Real y);
extern Real MaxReal(const Real x, const Real y);
extern int EqualReal(const Real x, const Real y);
extern int MinInt(const int x, const int y);
extern int MaxInt(const int x, const int y);
extern int Sign(const Real x);
extern Real Dot(const Real *RESTRICT V1, const Real *RESTRICT V2);
extern Real Norm(const Real *RESTRICT V);
extern Real Dist2(const Real *RESTRICT V1, const Real *RESTRICT V2);
extern Real Dist(const Real *RESTRICT V1, const Real *RESTRICT V2);
extern void Cross(const Real *RESTRICT V1, const Real *RESTRICT V2, Real *RESTRICT V);
extern void OrthogonalSpace(const Real *RESTRICT N, Real *RESTRICT Ta, Real *RESTRICT Tb);
extern void Normalize(const int dimV, const Real normalizer, Real *RESTRICT V);

#ifdef __cplusplus
}
#endif
#endif