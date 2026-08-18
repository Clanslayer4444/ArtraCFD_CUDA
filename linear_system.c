/****************************************************************************
 * Required Header Files
 ****************************************************************************/
#include "linear_system.h"
#include <stdio.h>
#include <math.h>
#include "commons.h"

/****************************************************************************
 * Static Function Declarations
 ****************************************************************************/
static int LUFactorization(const int n, Real * RESTRICT A, int permute[RESTRICT]);
static int SolveFactorizedLinearSystem(const int n, Real * RESTRICT L, Real * RESTRICT U,
        Real x[], Real b[], const int permute[RESTRICT]);

/****************************************************************************
 * Function definitions
 ****************************************************************************/
int SolveLinearSystem(const int n, Real * RESTRICT A,
        const int m, Real * RESTRICT X, Real * RESTRICT B)
{
    int permute[n]; /* record the permutation information */
    Real rhs[n];    /* transfer data into column vector */

    /* Cast pointers back to 2D arrays for indexing */
    Real (*AA)[n] = (Real (*)[n]) A;
    Real (*XX)[m] = (Real (*)[m]) X;
    Real (*BB)[m] = (Real (*)[m]) B;

    LUFactorization(n, A, permute);

    for (int col = 0; col < m; ++col) {
        for (int row = 0; row < n; ++row) {
            rhs[row] = BB[row][col]; /* obtain RHS vector */
        }
        SolveFactorizedLinearSystem(n, A, A, rhs, rhs, permute);
        for (int row = 0; row < n; ++row) {
            XX[row][col] = rhs[row]; /* save solution vector */
        }
    }
    return 0;
}

/*
 * Perform A = LU factorization
 */
static int LUFactorization(const int n, Real * RESTRICT A, int permute[RESTRICT])
{
    Real (*AA)[n] = (Real (*)[n]) A;

    const Real epsilon = 1.0e-15;
    const Real zero = 0.0;
    const Real one = 1.0;
    Real temp = 0.0;
    Real maximum = 0.0;
    Real scale[n];
    int rowMax = 0;
    int sign = 1;

    for (int row = 0; row < n; ++row) {
        maximum = zero;
        for (int col = 0; col < n; ++col) {
            temp = fabs(AA[row][col]);
            if (temp > maximum) maximum = temp;
        }
        if (zero == maximum) {
            ShowError("singular matrix in LU factorization...");
        }
        scale[row] = one / maximum;
    }

    for (int loop = 0; loop < n; ++loop) {
        maximum = zero;
        rowMax = loop;
        for (int row = loop; row < n; ++row) {
            temp = scale[row] * fabs(AA[row][loop]);
            if (temp > maximum) {
                maximum = temp;
                rowMax = row;
            }
        }
        if (loop != rowMax) {
            for (int col = 0; col < n; ++col) {
                temp = AA[rowMax][col];
                AA[rowMax][col] = AA[loop][col];
                AA[loop][col] = temp;
            }
            sign = -sign;
            scale[rowMax] = scale[loop];
        }
        permute[loop] = rowMax;
        if (zero == AA[loop][loop]) {
            AA[loop][loop] = epsilon;
        }
        for (int row = loop + 1; row < n; ++row) {
            AA[row][loop] = AA[row][loop] / AA[loop][loop];
            for (int col = loop + 1; col < n; ++col) {
                AA[row][col] = AA[row][col] - AA[row][loop] * AA[loop][col];
            }
        }
    }
    return sign;
}

/*
 * Solve LU * x = b
 */
static int SolveFactorizedLinearSystem(const int n, Real * RESTRICT L, Real * RESTRICT U,
        Real x[], Real b[], const int permute[RESTRICT])
{
    Real (*LL)[n] = (Real (*)[n]) L;
    Real (*UU)[n] = (Real (*)[n]) U;

    Real temp = 0.0;
    for (int row = 0; row < n; ++row) {
        temp = b[row];
        x[row] = b[permute[row]];
        b[permute[row]] = temp;
    }

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < row; ++col) {
            x[row] = x[row] - LL[row][col] * x[col];
        }
    }

    for (int row = n - 1; row >= 0; --row) {
        for (int col = row + 1; col < n; ++col) {
            x[row] = x[row] - UU[row][col] * x[col];
        }
        x[row] = x[row] / UU[row][row];
    }
    return 0;
}
