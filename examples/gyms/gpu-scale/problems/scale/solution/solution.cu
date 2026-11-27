#include <cuda_runtime.h>

__global__ void scale(const float* X, float* Y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Y[i] = 2.0f * X[i];
    }
}

extern "C" void solve(const float* X, float* Y, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    scale<<<blocks, threads>>>(X, Y, N);
    cudaDeviceSynchronize();
}
