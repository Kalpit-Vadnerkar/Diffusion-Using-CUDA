#include "gl_display.h"
#include "cpu_processor.h"
#include "cuda_processor.h"
#include <GL/glut.h>

#define WIDTH 512
#define HEIGHT 512
#define THREADS_PER_BLOCK 16

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutInitWindowPosition(100, 50);
    glutCreateWindow("CUDA/CPU Diffusion Compare");
    
    // Initialize both GPU and CPU
    initCuda(WIDTH, HEIGHT, 16);
    init_cpu_buffers(WIDTH, HEIGHT);
    
    setup_the_view(WIDTH, HEIGHT);
    
    // Print usage instructions
    print_usage();
    
    glutDisplayFunc(draw_stuff);
    glutKeyboardFunc(getout);
    glutReshapeFunc(reshape);
    
    glutMainLoop();
    return 0;
}