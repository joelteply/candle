// Regression: pre-Volta FP16 reductions need the compatibility atomicAdd.
// Exercise both halves of one CAS word concurrently, preserve the adjacent
// sentinel, and prove NaN does not make the integer CAS retry loop hang.
// Build with nvcc -arch=sm_61; run on a Pascal device (or its PTX-compatible GPU).
#include "../src/compatibility.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

__global__ void add_halves(__half* values) {
    atomicAdd(values + (threadIdx.x & 1), __float2half(1.0f));
    atomicAdd(values + 2, __float2half(1.0f));
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 1;
    for (int device = 0; device < count; ++device) {
        if (cudaSetDevice(device) != cudaSuccess) return 2;
        __half values[] = {__float2half(0), __float2half(0),
                           __float2half(NAN), __float2half(19)};
        __half* gpu = nullptr;
        if (cudaMalloc(&gpu, sizeof(values)) != cudaSuccess) return 3;
        if (cudaMemcpy(gpu, values, sizeof(values), cudaMemcpyHostToDevice) != cudaSuccess) return 4;
        add_halves<<<1, 64>>>(gpu);
        if (cudaDeviceSynchronize() != cudaSuccess) return 5;
        if (cudaMemcpy(values, gpu, sizeof(values), cudaMemcpyDeviceToHost) != cudaSuccess) return 6;
        if (cudaFree(gpu) != cudaSuccess) return 7;
        if (__half2float(values[0]) != 32 || __half2float(values[1]) != 32 ||
            !std::isnan(__half2float(values[2])) || __half2float(values[3]) != 19) return 8;
        std::printf("PASS: device %d half atomic contention, neighbor preservation, NaN termination\n", device);
    }
    return 0;
}
