// cpu_processor.c
#include "cpu_processor.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <stdio.h>

static float* image = NULL;
static float* processed_image = NULL;
static int width = 0;
static int height = 0;
static const float beta = 0.1f;

float generate_gaussian_noise() {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    
    // Prevent log(0)
    if (u1 < 1e-7f) u1 = 1e-7f;
    
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

void print_processing_time(const char* operation, float milliseconds) {
    printf("CPU %s execution time: %.3f ms\n", operation, milliseconds);
}

void init_cpu_buffers(int w, int h) {
    width = w;
    height = h;
    int size = width * height * 3;
    
    image = (float*)malloc(size * sizeof(float));
    processed_image = (float*)malloc(size * sizeof(float));
    
    srand(time(NULL));  // Initialize random seed
    
    // Initialize with same checkered pattern as GPU
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            int checker = ((x / 32) + (y / 32)) % 2;
            float value = checker ? 1.0f : 0.0f;
            
            image[idx] = value;     // R
            image[idx + 1] = value; // G
            image[idx + 2] = value; // B
        }
    }
}

void process_forward_diffusion_cpu() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    int size = width * height * 3;
    for (int i = 0; i < size; i++) {
        float pixel = image[i];
        float noise = generate_gaussian_noise();
        processed_image[i] = sqrtf(1.0f - beta) * pixel + sqrtf(beta) * noise;
    }
    
    // Swap buffers
    float* temp = image;
    image = processed_image;
    processed_image = temp;
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    print_processing_time("Forward Diffusion", milliseconds);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void process_reverse_diffusion_cpu() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    int size = width * height * 3;
    for (int i = 0; i < size; i++) {
        float noisy_pixel = image[i];
        processed_image[i] = (noisy_pixel - sqrtf(beta)) / sqrtf(1.0f - beta);
    }
    
    // Swap buffers
    float* temp = image;
    image = processed_image;
    processed_image = temp;
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    print_processing_time("Reverse Diffusion", milliseconds);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

float* get_cpu_result() {
    return image;
}

void cleanup_cpu_buffers() {
    free(image);
    free(processed_image);
    image = NULL;
    processed_image = NULL;
}