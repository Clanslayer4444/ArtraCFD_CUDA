#ifndef GPU_EIGEN_DEVICE_H
#define GPU_EIGEN_DEVICE_H

#include "commons.h"

__device__ __forceinline__
void Eigenvalue_dev(int s, const Real *Uo, Real *Lambda)
{
    Real u = Uo[s + 1];
    Real c = Uo[5];

    Lambda[0] = u - c;
    Lambda[1] = u;
    Lambda[2] = u;
    Lambda[3] = u;
    Lambda[4] = u + c;
}

__device__ __forceinline__
void EigenvectorL_dev(int s, Real gamma, const Real *Uo, Real *L)
{
    const Real u = Uo[1];
    const Real v = Uo[2];
    const Real w = Uo[3];
    const Real c = Uo[5];
    const Real q = 0.5 * (u*u + v*v + w*w);
    const Real b = (gamma - 1.0) / (2.0 * c * c);
    const Real d = 1.0 / (2.0 * c);


    if (s == X) {
        L[0*DIMU+0] = b*q + d*u;    L[0*DIMU+1] = -b*u - d;   L[0*DIMU+2] = -b*v;
        L[0*DIMU+3] = -b*w;         L[0*DIMU+4] = b;

        L[1*DIMU+0] = -2*b*q + 1;   L[1*DIMU+1] = 2*b*u;     L[1*DIMU+2] = 2*b*v;
        L[1*DIMU+3] = 2*b*w;        L[1*DIMU+4] = -2*b;

        L[2*DIMU+0] = -2*b*q*v;     L[2*DIMU+1] = 2*b*v*u;   L[2*DIMU+2] = 2*b*v*v + 1;
        L[2*DIMU+3] = 2*b*w*v;      L[2*DIMU+4] = -2*b*v;

        L[3*DIMU+0] = -2*b*q*w;     L[3*DIMU+1] = 2*b*w*u;   L[3*DIMU+2] = 2*b*w*v;
        L[3*DIMU+3] = 2*b*w*w + 1;  L[3*DIMU+4] = -2*b*w;

        L[4*DIMU+0] = b*q - d*u;    L[4*DIMU+1] = -b*u + d;  L[4*DIMU+2] = -b*v;
        L[4*DIMU+3] = -b*w;         L[4*DIMU+4] = b;
    }
    else if (s == Y) {
        L[0*DIMU+0] = b*q + d*v;    L[0*DIMU+1] = -b*u;       L[0*DIMU+2] = -b*v - d;
        L[0*DIMU+3] = -b*w;         L[0*DIMU+4] = b;

        L[1*DIMU+0] = -2*b*q*u;     L[1*DIMU+1] = 2*b*u*u+1; L[1*DIMU+2] = 2*b*v*u;
        L[1*DIMU+3] = 2*b*w*u;      L[1*DIMU+4] = -2*b*u;

        L[2*DIMU+0] = -2*b*q + 1;   L[2*DIMU+1] = 2*b*u;     L[2*DIMU+2] = 2*b*v;
        L[2*DIMU+3] = 2*b*w;        L[2*DIMU+4] = -2*b;

        L[3*DIMU+0] = -2*b*q*w;     L[3*DIMU+1] = 2*b*w*u;   L[3*DIMU+2] = 2*b*w*v;
        L[3*DIMU+3] = 2*b*w*w+1;    L[3*DIMU+4] = -2*b*w;

        L[4*DIMU+0] = b*q - d*v;    L[4*DIMU+1] = -b*u;      L[4*DIMU+2] = -b*v + d;
        L[4*DIMU+3] = -b*w;         L[4*DIMU+4] = b;
    }
    else { // Z
        L[0*DIMU+0] = b*q + d*w;    L[0*DIMU+1] = -b*u;       L[0*DIMU+2] = -b*v;
        L[0*DIMU+3] = -b*w - d;     L[0*DIMU+4] = b;

        L[1*DIMU+0] = -2*b*q*u;     L[1*DIMU+1] = 2*b*u*u+1; L[1*DIMU+2] = 2*b*v*u;
        L[1*DIMU+3] = 2*b*w*u;      L[1*DIMU+4] = -2*b*u;

        L[2*DIMU+0] = -2*b*q*v;     L[2*DIMU+1] = 2*b*v*u;   L[2*DIMU+2] = 2*b*v*v+1;
        L[2*DIMU+3] = 2*b*w*v;      L[2*DIMU+4] = -2*b*v;

        L[3*DIMU+0] = -2*b*q + 1;   L[3*DIMU+1] = 2*b*u;     L[3*DIMU+2] = 2*b*v;
        L[3*DIMU+3] = 2*b*w;        L[3*DIMU+4] = -2*b;

        L[4*DIMU+0] = b*q - d*w;    L[4*DIMU+1] = -b*u;      L[4*DIMU+2] = -b*v;
        L[4*DIMU+3] = -b*w + d;     L[4*DIMU+4] = b;
    }

    
}

__device__ __forceinline__
void EigenvectorR_dev(int s, const Real *Uo, Real *R)
{
    
    
    const Real u  = Uo[1];
    const Real v  = Uo[2];
    const Real w  = Uo[3];
    const Real hT = Uo[4];
    const Real c  = Uo[5];
    const Real q  = 0.5 * (u*u + v*v + w*w);

    if (s == X) {
        R[0*DIMU + 0] = 1.0;       R[0*DIMU + 1] = 1.0;       R[0*DIMU + 2] = 0.0;  R[0*DIMU + 3] = 0.0;  R[0*DIMU + 4] = 1.0;
        R[1*DIMU + 0] = u - c;     R[1*DIMU + 1] = u;         R[1*DIMU + 2] = 0.0;  R[1*DIMU + 3] = 0.0;  R[1*DIMU + 4] = u + c;
        R[2*DIMU + 0] = v;         R[2*DIMU + 1] = 0.0;       R[2*DIMU + 2] = 1.0;  R[2*DIMU + 3] = 0.0;  R[2*DIMU + 4] = v;
        R[3*DIMU + 0] = w;         R[3*DIMU + 1] = 0.0;       R[3*DIMU + 2] = 0.0;  R[3*DIMU + 3] = 1.0;  R[3*DIMU + 4] = w;
        R[4*DIMU + 0] = hT - u*c;  R[4*DIMU + 1] = u*u - q;   R[4*DIMU + 2] = v;    R[4*DIMU + 3] = w;    R[4*DIMU + 4] = hT + u*c;
    }
    else if (s == Y) {
        R[0*DIMU + 0] = 1.0;       R[0*DIMU + 1] = 0.0;  R[0*DIMU + 2] = 1.0;       R[0*DIMU + 3] = 0.0;  R[0*DIMU + 4] = 1.0;
        R[1*DIMU + 0] = u;         R[1*DIMU + 1] = 1.0;  R[1*DIMU + 2] = 0.0;       R[1*DIMU + 3] = 0.0;  R[1*DIMU + 4] = u;
        R[2*DIMU + 0] = v - c;     R[2*DIMU + 1] = 0.0;  R[2*DIMU + 2] = v;         R[2*DIMU + 3] = 0.0;  R[2*DIMU + 4] = v + c;
        R[3*DIMU + 0] = w;         R[3*DIMU + 1] = 0.0;  R[3*DIMU + 2] = 0.0;       R[3*DIMU + 3] = 1.0;  R[3*DIMU + 4] = w;
        R[4*DIMU + 0] = hT - v*c;  R[4*DIMU + 1] = u;    R[4*DIMU + 2] = v*v - q;   R[4*DIMU + 3] = w;    R[4*DIMU + 4] = hT + v*c;
    }
    else { // Z
        R[0*DIMU + 0] = 1.0;       R[0*DIMU + 1] = 0.0;  R[0*DIMU + 2] = 0.0;  R[0*DIMU + 3] = 1.0;       R[0*DIMU + 4] = 1.0;
        R[1*DIMU + 0] = u;         R[1*DIMU + 1] = 1.0;  R[1*DIMU + 2] = 0.0;  R[1*DIMU + 3] = 0.0;       R[1*DIMU + 4] = u;
        R[2*DIMU + 0] = v;         R[2*DIMU + 1] = 0.0;  R[2*DIMU + 2] = 1.0;  R[2*DIMU + 3] = 0.0;       R[2*DIMU + 4] = v;
        R[3*DIMU + 0] = w - c;     R[3*DIMU + 1] = 0.0;  R[3*DIMU + 2] = 0.0;  R[3*DIMU + 3] = w;         R[3*DIMU + 4] = w + c;
        R[4*DIMU + 0] = hT - w*c;  R[4*DIMU + 1] = u;    R[4*DIMU + 2] = v;    R[4*DIMU + 3] = w*w - q;  R[4*DIMU + 4] = hT + w*c;
    }
    
    
    
    
}

#endif
