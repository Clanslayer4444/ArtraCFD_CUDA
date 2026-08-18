#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello_kernel() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    // Allocate some GPU memory
    int *d_data;
    cudaMalloc((void**)&d_data, 100 * sizeof(int));
    cudaFree(d_data);

    // Launch a kernel
    hello_kernel<<<1, 4>>>();
    cudaDeviceSynchronize();

    printf("GPU allocation and kernel launch worked!\n");
    return 0;
}
