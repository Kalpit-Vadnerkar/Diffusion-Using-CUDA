// cuda_processor.cu
#include "cuda_processor.h"
#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

static int width, height;
static int threads_per_block;
static float *d_image = NULL;
static float *d_noisy_image = NULL;
static float *d_processed_image = NULL;
static float *h_image = NULL;
static curandState *d_states = NULL;
static const float beta = 0.1f;

// Get device properties for maximum threads per block
int get_max_threads_per_block() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return (int)sqrt(prop.maxThreadsPerBlock);  // Square root since we use square blocks
}

int get_current_threads_per_block() {
    return threads_per_block;
}

void calculate_grid_block_dim(int width, int height, int threads_per_block, 
                            int* num_blocks_x, int* num_blocks_y) {
    *num_blocks_x = (width + threads_per_block - 1) / threads_per_block;
    *num_blocks_y = (height + threads_per_block - 1) / threads_per_block;
}

void print_kernel_time(const char* kernel_name, float milliseconds) {
    printf("'%s' Kernel execution time: %.3f ms\n", kernel_name, milliseconds);
}

void initCuda(int w, int h, int tpb) {
    width = w;
    height = h;
    threads_per_block = tpb;
    
    int size = width * height * 3 * sizeof(float);
    
    cudaMalloc(&d_image, size);
    cudaMalloc(&d_noisy_image, size);
    cudaMalloc(&d_processed_image, size);
    cudaMalloc(&d_states, width * height * sizeof(curandState));
    h_image = (float*)malloc(size);
    
    int num_blocks_x, num_blocks_y;
    calculate_grid_block_dim(width, height, threads_per_block, &num_blocks_x, &num_blocks_y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    dim3 blocks(num_blocks_x, num_blocks_y);
    dim3 threads(threads_per_block, threads_per_block);
    initializeImageKernel<<<blocks, threads>>>(d_image, width, height);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    print_kernel_time("GPU Initialize Image", milliseconds);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    download_to_host();
}

void resize_image(int new_width, int new_height) {
    cuda_cleanup();
    initCuda(new_width, new_height, threads_per_block);
}

void update_threads_per_block(int threads) {
    threads_per_block = threads;
    // Reinitialize with new thread configuration
    resize_image(width, height);
}

void process_forward_diffusion() {
    int num_blocks_x, num_blocks_y;
    calculate_grid_block_dim(width, height, threads_per_block, &num_blocks_x, &num_blocks_y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    dim3 blocks(num_blocks_x, num_blocks_y);
    dim3 threads(threads_per_block, threads_per_block);
    forwardDiffusionKernel<<<blocks, threads>>>(
        d_image,
        d_noisy_image,
        beta,
        width,
        height,
        d_states
    );
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    print_kernel_time("GPU Forward Diffusion", milliseconds);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    float* temp = d_image;
    d_image = d_noisy_image;
    d_noisy_image = temp;
    
    download_to_host();
}

void process_forward_diffusion_shared() {
    int num_blocks_x, num_blocks_y;
    calculate_grid_block_dim(width, height, threads_per_block, &num_blocks_x, &num_blocks_y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    dim3 blocks(num_blocks_x, num_blocks_y);
    dim3 threads(threads_per_block, threads_per_block);
    size_t shared_mem_size = threads_per_block * threads_per_block * 3 * sizeof(float);
    forwardDiffusionSharedKernel<<<blocks, threads, shared_mem_size>>>(
        d_image,
        d_noisy_image,
        beta,
        width,
        height,
        d_states
    );
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    print_kernel_time("GPU Forward Diffusion Shared", milliseconds);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    float* temp = d_image;
    d_image = d_noisy_image;
    d_noisy_image = temp;
    
    download_to_host();
}

void process_reverse_diffusion() {
    int num_blocks_x, num_blocks_y;
    calculate_grid_block_dim(width, height, threads_per_block, &num_blocks_x, &num_blocks_y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    dim3 blocks(num_blocks_x, num_blocks_y);
    dim3 threads(threads_per_block, threads_per_block);
    reverseDiffusionKernel<<<blocks, threads>>>(
        d_image,
        d_processed_image,
        beta,
        width,
        height
    );
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    print_kernel_time("GPU Reverse Diffusion", milliseconds);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    float* temp = d_image;
    d_image = d_processed_image;
    d_processed_image = temp;
    
    download_to_host();
}

void upload_to_device(float* host_data) {
    cudaMemcpy(d_image, host_data, width * height * 3 * sizeof(float), cudaMemcpyHostToDevice);
}

void download_to_host() {
    cudaMemcpy(h_image, d_image, width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost);
}

void cuda_cleanup() {
    if (d_image) cudaFree(d_image);
    if (d_noisy_image) cudaFree(d_noisy_image);
    if (d_processed_image) cudaFree(d_processed_image);
    if (d_states) cudaFree(d_states);
    if (h_image) free(h_image);
}

float* get_host_image() {
    return h_image;
}

int get_width() {
    return width;
}

int get_height() {
    return height;
}