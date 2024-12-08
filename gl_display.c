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
    
    switch(key) {
        case 'i':
            cuda_cleanup();
            initCuda(get_width(), get_height(), threads);
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
        default:
            break;
    }
}