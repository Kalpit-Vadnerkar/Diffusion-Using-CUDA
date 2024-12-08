CC = gcc
NVCC = nvcc
NVCCFLAGS = -allow-unsupported-compiler
LDFLAGS = -lGL -lGLU -lglut

# Headers
HEADERS = cuda_processor.h cuda_kernels.cuh gl_display.h cpu_processor.h

# Source files
CUDA_SOURCES = cuda_processor.cu cuda_kernels.cu cpu_processor.cu
C_SOURCES = main.c gl_display.c

# Object files
OBJECTS = $(CUDA_SOURCES:.cu=.o) $(C_SOURCES:.c=.o)

# Executable name
TARGET = diffusion_viewer

# Default target
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.c $(HEADERS)
	$(CC) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean