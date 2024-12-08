#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Initialize the image with a pattern
__global__ void initializeImageKernel(float* d_image, int width, int height);

// Forward diffusion (adding noise)
__global__ void forwardDiffusionKernel(
    float* d_image,
    float* d_noisy_image,
    float beta,
    int width,
    int height,
    curandState* states
);

__global__ void forwardDiffusionSharedKernel(
    float* d_image,
    float* d_noisy_image,
    float beta,
    int width,
    int height,
    curandState* states
);

// Reverse diffusion (denoising)
__global__ void reverseDiffusionKernel(
    float* d_noisy_image,
    float* d_denoised_image,
    float beta,
    int width,
    int height
);

#endif // CUDA_KERNELS_CUH