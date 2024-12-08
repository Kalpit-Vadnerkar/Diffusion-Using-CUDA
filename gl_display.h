#ifndef GL_DISPLAY_H
#define GL_DISPLAY_H

void setup_the_view(int width, int height);
void reshape(int w, int h); 
void resize_window(int width, int height);
void draw_stuff(void);
void cleanup(void);
void getout(unsigned char key, int x, int y);
void print_usage(void);

#endif // GL_DISPLAY_H