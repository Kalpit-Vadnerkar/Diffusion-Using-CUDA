#include "cuda_kernels.cuh"

__global__ void generateCheckerPattern(float* d_pattern, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        // Create a checkered pattern
        int checker = ((x / 32) + (y / 32)) % 2;
        float value = checker ? 1.0f : 0.0f;
        
        d_pattern[idx] = value;     // R
        d_pattern[idx + 1] = value; // G
        d_pattern[idx + 2] = value; // B
    }
}

__global__ void forwardDiffusionKernel(
    float* d_image,
    float* d_noisy_image,
    float beta,
    int width,
    int height,
    curandState* states
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        int thread_idx = y * width + x;
        
        // Initialize random state for this thread
        curand_init(clock64(), thread_idx, 0, &states[thread_idx]);
        
        // Process each color channel
        for (int c = 0; c < 3; c++) {
            float pixel = d_image[idx + c];
            float noise = curand_normal(&states[thread_idx]);
            d_noisy_image[idx + c] = sqrt(1.0f - beta) * pixel + sqrt(beta) * noise;
        }
    }
}

__global__ void forwardDiffusionSharedKernel(
    float* d_image,
    float* d_noisy_image,
    float beta,
    int width,
    int height,
    curandState* states
) {
    extern __shared__ float shared_pixels[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int x = bx + tx;
    int y = by + ty;
    
    // Calculate indices
    int block_idx = ty * blockDim.x + tx;
    int shared_idx = block_idx * 3;  // 3 channels per pixel
    int global_idx = (y * width + x) * 3;
    
    // Load data into shared memory
    if (x < width && y < height) {
        shared_pixels[shared_idx] = d_image[global_idx];
        shared_pixels[shared_idx + 1] = d_image[global_idx + 1];
        shared_pixels[shared_idx + 2] = d_image[global_idx + 2];
    }
    
    __syncthreads();
    
    if (x < width && y < height) {
        // Initialize random state for this thread
        int thread_idx = y * width + x;
        curand_init(clock64(), thread_idx, 0, &states[thread_idx]);
        
        // Process each color channel
        for (int c = 0; c < 3; c++) {
            float pixel = shared_pixels[shared_idx + c];
            float noise = curand_normal(&states[thread_idx]);
            d_noisy_image[global_idx + c] = sqrt(1.0f - beta) * pixel + sqrt(beta) * noise;
        }
    }
}

__global__ void reverseDiffusionKernel(
    float* d_noisy_image,
    float* d_denoised_image,
    float beta,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        
        // Process each color channel
        for (int c = 0; c < 3; c++) {
            float noisy_pixel = d_noisy_image[idx + c];
            d_denoised_image[idx + c] = (noisy_pixel - sqrt(beta)) / sqrt(1.0f - beta);
        }
    }
}

__global__ void computeErrorKernel(
    float* d_current_image,
    float* d_reference_image,
    float* d_error,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        float sum_squared_error = 0.0f;
        
        // Compute squared error for each channel
        for (int c = 0; c < 3; c++) {
            float diff = d_current_image[idx + c] - d_reference_image[idx + c];
            sum_squared_error += diff * diff;
        }
        
        // Store average error for this pixel
        d_error[y * width + x] = sum_squared_error / 3.0f;
    }
}
