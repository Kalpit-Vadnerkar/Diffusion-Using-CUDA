#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void generateCheckerPattern(float* d_pattern, int width, int height);

// Compute error between current image and reference
__global__ void computeErrorKernel(
    float* d_current_image,
    float* d_reference_image,
    float* d_error,
    int width,
    int height
);

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