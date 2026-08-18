#include "weno.h"
#include "cfd_commons.h"
#include "commons.h"

static Real Square(const Real);

void WENO5(Real F[RESTRICT][DIMU], Real Fhat[RESTRICT])
{
    /* Explicitly map the 5 stencil points to local variables for safety */
    /* F is expected to point to start of stencil. F[2] is center. */
    /* Accessing F[0]..F[4] */
    
    Real omega[3], q[3], IS[3], alpha[3];
    const Real C[3] = {0.1, 0.6, 0.3};
    const Real epsilon = 1.0e-6;

    for (int r = 0; r < DIMU; ++r) {
        Real f0 = F[0][r];
        Real f1 = F[1][r];
        Real f2 = F[2][r]; /* Center */
        Real f3 = F[3][r];
        Real f4 = F[4][r];

        IS[0] = (13.0/12.0)*Square(f0 - 2.0*f1 + f2) + 0.25*Square(f0 - 4.0*f1 + 3.0*f2);
        IS[1] = (13.0/12.0)*Square(f1 - 2.0*f2 + f3) + 0.25*Square(f1 - f3);
        IS[2] = (13.0/12.0)*Square(f2 - 2.0*f3 + f4) + 0.25*Square(3.0*f2 - 4.0*f3 + f4);

        alpha[0] = C[0] / Square(epsilon + IS[0]);
        alpha[1] = C[1] / Square(epsilon + IS[1]);
        alpha[2] = C[2] / Square(epsilon + IS[2]);
        
        Real sum = alpha[0] + alpha[1] + alpha[2];
        omega[0] = alpha[0]/sum; omega[1] = alpha[1]/sum; omega[2] = alpha[2]/sum;

        q[0] = (1.0/6.0)*(2.0*f0 - 7.0*f1 + 11.0*f2);
        q[1] = (1.0/6.0)*(-f1 + 5.0*f2 + 2.0*f3);
        q[2] = (1.0/6.0)*(2.0*f2 + 5.0*f3 - f4);

        Fhat[r] = omega[0]*q[0] + omega[1]*q[1] + omega[2]*q[2];
    }
}
static Real Square(const Real x) { return x * x; }