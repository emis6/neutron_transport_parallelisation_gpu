SRCS = neutron-omp.cu
EXE_NAME = neutron-omp
OBJECTS = neutron-omp.o

CC = gcc
CFLAGS = -O3 -arch=sm_20 -lineinfo   #-std=c11
LIB=-lm -L/usr/local/cuda/lib64/ -lcuda -lcudart #-openmp
NVCC= /usr/local/cuda/bin/nvcc -Xcompiler -fopenmp


all: ${EXE_NAME}

neutron-omp.o : neutron-omp.cu
	$(NVCC) -c $(CFLAGS) $< 

neutron-omp : neutron-omp.o
	${NVCC} ${CFLAGS} -o $@ $+ ${LIB}

clean:
	rm -f ${EXE_NAME} *.o *~
