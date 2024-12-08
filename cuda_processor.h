// cuda_processor.h
#ifndef CUDA_PROCESSOR_H
#define CUDA_PROCESSOR_H

#ifdef __cplusplus
extern "C" {
#endif

void initCuda(int width, int height, int threads_per_block);
void generate_reference_pattern(void);
void reset_to_original(void);
float compute_error(void);
void cuda_cleanup(void);
float* get_host_image(void);
void download_to_host(void);
void process_forward_diffusion(void);
void process_forward_diffusion_shared(void);
void process_reverse_diffusion(void);
int get_width(void);
int get_height(void);
void resize_image(int new_width, int new_height);
void update_threads_per_block(int threads);
int get_max_threads_per_block(void);
int get_current_threads_per_block(void);
void set_beta(float new_beta);
float get_beta(void);

#ifdef __cplusplus
}
#endif

#endif // CUDA_PROCESSOR_H