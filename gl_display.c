#include "gl_display.h"
#include "cpu_processor.h"
#include "cuda_processor.h"
#include <GL/gl.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>

void setup_the_view(int width, int height) {
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width, 0, height, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void reshape(int w, int h) {
    setup_the_view(w, h);
}

void resize_window(int width, int height) {
    // Resize the window
    glutReshapeWindow(width, height);
    
    // Update the viewport and projection
    setup_the_view(width, height);
    
    // Force redisplay
    glutPostRedisplay();
}

void draw_stuff() {
    glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2i(0, 0);
    glDisable(GL_DEPTH_TEST);
    
    // Draw the image processed by CUDA using GL_FLOAT
    glDrawPixels(get_width(), get_height(), GL_RGB, GL_FLOAT, get_host_image());
    
    glutSwapBuffers();
}

void cleanup() {
    cuda_cleanup();
    exit(0);
}

void getout(unsigned char key, int x, int y) {
    static int current_size = 512;
    static int threads = 16;
    int max_threads = get_max_threads_per_block();
    float current_beta;
    float beta_step = 0.05f;  // Amount to change beta by
    
    switch(key) {
        case 'i':
            reset_to_original();  // Replace initCuda call with reset_to_original
            glutPostRedisplay();
            break;
        case 'q':
            cleanup_cpu_buffers();
            cleanup();
            break;
        case '[':  // Decrease image size
            if (current_size > 256) {
                current_size -= 256;
                resize_image(current_size, current_size);
                resize_window(current_size, current_size);
                init_cpu_buffers(current_size, current_size);  // Initialize CPU buffers too
                printf("Image size: %dx%d\n", current_size, current_size);
            }
            break;
        case ']':  // Increase image size
            if (current_size < 4080) {
                current_size += 256;
                resize_image(current_size, current_size);
                resize_window(current_size, current_size);
                init_cpu_buffers(current_size, current_size);  // Initialize CPU buffers too
                printf("Image size: %dx%d\n", current_size, current_size);
            }
            break;
        case '-':  // Decrease threads per block
            if (threads > 4) {
                threads -= 4;
                update_threads_per_block(threads);
                printf("Threads per block: %dx%d\n", threads, threads);
                glutPostRedisplay();
            }
            break;
        case '=':  // Increase threads per block
            if (threads < max_threads - 4) {
                threads += 4;
                update_threads_per_block(threads);
                printf("Threads per block: %dx%d\n", threads, threads);
                glutPostRedisplay();
            }
            break;
        case 'F':  
            process_forward_diffusion_cpu();
            glDrawPixels(get_width(), get_height(), GL_RGB, GL_FLOAT, get_cpu_result());
            glutSwapBuffers();
            break;
        case 'R':  
            process_reverse_diffusion_cpu();
            glDrawPixels(get_width(), get_height(), GL_RGB, GL_FLOAT, get_cpu_result());
            glutSwapBuffers();
            break;  
        case 'f':
            process_forward_diffusion();
            glutPostRedisplay();
            break;
        case 's':
            process_forward_diffusion_shared();
            glutPostRedisplay();
            break;
        case 'r':
            process_reverse_diffusion();
            glutPostRedisplay();
            break;
        case ',':  // Decrease beta
            current_beta = get_beta();
            set_beta(current_beta - beta_step);
            set_cpu_beta(current_beta - beta_step);
            glutPostRedisplay();
            break;
        case '.':  // Increase beta
            current_beta = get_beta();
            set_beta(current_beta + beta_step);
            set_cpu_beta(current_beta + beta_step);
            glutPostRedisplay();
            break;
        default:
            break;
    }
}

void print_usage() {
    printf("\nDiffusion Viewer Usage Commands:\n");
    printf("--------------------------------\n");
    printf("Image Processing:\n");
    printf("  f - Run GPU forward diffusion\n");
    printf("  s - Run GPU forward diffusion with shared memory\n");
    printf("  r - Run GPU reverse diffusion\n");
    printf("  F - Run CPU forward diffusion\n");
    printf("  R - Run CPU reverse diffusion\n");
    printf("\nImage Controls:\n");
    printf("  i - Reset image to original pattern\n");
    printf("  [ - Decrease image size by 256 pixels (min: 256x256)\n");
    printf("  ] - Increase image size by 256 pixels (max: 4080x4080)\n");
    printf("  , - Decrease beta value (noise factor)\n");
    printf("  . - Increase beta value (noise factor)\n");
    printf("\nPerformance Controls:\n");
    printf("  - - Decrease threads per block by 4\n");
    printf("  = - Increase threads per block by 4\n");
    printf("\nApplication Control:\n");
    printf("  q - Quit application\n");
    printf("\nCurrent Settings:\n");
    printf("  Image Size: %dx%d\n", get_width(), get_height());
    printf("  Threads per Block: %dx%d\n", get_current_threads_per_block(), get_current_threads_per_block());
    printf("  Beta Value: %.3f\n", get_beta());
    printf("\n");
}
