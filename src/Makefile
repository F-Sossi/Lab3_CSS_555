
SOURCES = kernel.cu
OBJECTS = kernel.o

all: kernel.o lab3

CC      = nvcc
CFLAGS  = -I. -g -G -arch=sm_86 -lcublas kernel.cu
DEPS    = kernel.cu

%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $<

lab3: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm kernel.o lab3

