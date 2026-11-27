// GPU_GYM_STARTER
#include <cuda_runtime.h>

// X and Y are device pointers. N is the number of elements.
__global__ void scale(const float* X, float* Y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
    }
}

extern "C" void solve(const float* X, float* Y, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    scale<<<blocks, threads>>>(X, Y, N);
    cudaDeviceSynchronize();
}
