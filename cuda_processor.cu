// cuda_processor.cu
#include "cuda_processor.h"
#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

static int width, height;
static int threads_per_block;
static float* d_reference_pattern = NULL;
static float *d_image = NULL;
static float *d_noisy_image = NULL;
static float *d_processed_image = NULL;
static float *h_image = NULL;
static curandState *d_states = NULL;
static float beta = 0.1f;
static float* d_error = NULL;
static float* h_error = NULL;

float* get_host_image() {
    return h_image;
}

int get_width() {
    return width;
}

int get_height() {
    return height;
}

void upload_to_device(float* host_data) {
    cudaMemcpy(d_image, host_data, width * height * 3 * sizeof(float), cudaMemcpyHostToDevice);
}

void download_to_host() {
    cudaMemcpy(h_image, d_image, width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost);
}

void reset_to_original() {
    // Copy from reference pattern to d_image
    cudaMemcpy(d_image, d_reference_pattern, width * height * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
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

void set_beta(float new_beta) {
    if (new_beta > 0.0f && new_beta < 1.0f) {
        beta = new_beta;
        printf("Beta value updated to: %.3f\n", beta);
    }
}

float get_beta() {
    return beta;
}

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

void generate_reference_pattern() {
    int num_blocks_x = (width + threads_per_block - 1) / threads_per_block;
    int num_blocks_y = (height + threads_per_block - 1) / threads_per_block;
    
    dim3 blocks(num_blocks_x, num_blocks_y);
    dim3 threads(threads_per_block, threads_per_block);
    
    generateCheckerPattern<<<blocks, threads>>>(d_reference_pattern, width, height);
    cudaDeviceSynchronize();
}

void initCuda(int w, int h, int tpb) {
    width = w;
    height = h;
    threads_per_block = tpb;
    
    int size = width * height * 3 * sizeof(float);
    int error_size = width * height * sizeof(float);
    
    cudaMalloc(&d_image, size);
    cudaMalloc(&d_noisy_image, size);
    cudaMalloc(&d_processed_image, size);
    cudaMalloc(&d_states, width * height * sizeof(curandState));
    cudaMalloc(&d_error, error_size);
    cudaMalloc(&d_reference_pattern, size);  // Allocate memory for reference pattern
    h_image = (float*)malloc(size);
    h_error = (float*)malloc(error_size);
    
    // Generate initial pattern
    generate_reference_pattern();
    
    // Set initial image to pattern
    reset_to_original();
    
    download_to_host();
}

float compute_error() {
    int num_blocks_x = (width + threads_per_block - 1) / threads_per_block;
    int num_blocks_y = (height + threads_per_block - 1) / threads_per_block;
    
    dim3 blocks(num_blocks_x, num_blocks_y);
    dim3 threads(threads_per_block, threads_per_block);
    
    computeErrorKernel<<<blocks, threads>>>(
        d_image,
        d_reference_pattern,
        d_error,
        width,
        height
    );
    
    // Copy error data to host
    cudaMemcpy(h_error, d_error, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute average error
    float total_error = 0.0f;
    for (int i = 0; i < width * height; i++) {
        total_error += h_error[i];
    }
    
    float mse = total_error / (width * height);
    printf("Mean Squared Error: %.6f\n", mse);
    return mse;
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
    compute_error();
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
    compute_error(); 
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
    compute_error(); 
}

void cuda_cleanup() {
    if (d_image) cudaFree(d_image);
    if (d_noisy_image) cudaFree(d_noisy_image);
    if (d_processed_image) cudaFree(d_processed_image);
    if (d_states) cudaFree(d_states);
    if (d_error) cudaFree(d_error);
    if (d_reference_pattern) cudaFree(d_reference_pattern);
    if (h_image) free(h_image);
    if (h_error) free(h_error);
}

