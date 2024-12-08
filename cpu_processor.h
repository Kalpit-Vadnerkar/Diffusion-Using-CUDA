// cpu_processor.h
#ifndef CPU_PROCESSOR_H
#define CPU_PROCESSOR_H

#ifdef __cplusplus
extern "C" {
#endif

void init_cpu_buffers(int width, int height);
void cleanup_cpu_buffers(void);
void process_forward_diffusion_cpu(void);
void process_reverse_diffusion_cpu(void);
float* get_cpu_result(void);
void set_cpu_beta(float new_beta);
float get_cpu_beta(void);
void initialize_constant_memory(void);
void reset_to_original(void);
float compute_error(void);

#ifdef __cplusplus
}
#endif

#endif // CPU_PROCESSOR_H