void TestEigenvectorL(void)
{
    const int N = 1;
    const int s = X;
    const Real gamma = 1.4;

    Real Uo[DIMUo] = {
        1.0,   // rho
        2.5,   // u
        0.3,   // v
        0.1,   // w
        10.0,  // energy
        1.2    // c
    };

    Real Lambda_cpu[DIMU];
    Real L_cpu[DIMU][DIMU];

    Eigenvalue(s, Uo, Lambda_cpu);
    EigenvectorL(s, gamma, Uo, L_cpu);

    Real Lambda_gpu[DIMU];
    Real L_gpu[DIMU * DIMU];

    LaunchEigenTestGPU(
        N, s, gamma,
        Uo,
        Lambda_gpu,
        L_gpu
    );

    printf("\nEigenvector L validation:\n");
    for (int i = 0; i < DIMU; ++i) {
        for (int j = 0; j < DIMU; ++j) {
            Real diff = fabs(L_cpu[i][j] - L_gpu[i*DIMU + j]);
            printf("L[%d][%d]: CPU=%+.6e GPU=%+.6e diff=%+.2e\n",
                   i, j,
                   L_cpu[i][j],
                   L_gpu[i*DIMU + j],
                   diff);
        }
    }
}
