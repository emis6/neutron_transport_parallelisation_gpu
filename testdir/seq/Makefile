SRCS =  neutron-seq.c 
EXE_NAME =neutron-seq
OBJECTS = neutron-seq.o

CC = gcc
CFLAGS = -O3 -arch=sm_20 -lineinfo   #-std=c11
LIB=-lm -L/usr/local/cuda/lib64/ -lcuda -lcudart #-openmp
NVCC= /usr/local/cuda/bin/nvcc -Xcompiler -fopenmp


all: ${EXE_NAME}

neutron-seq.o : neutron-seq.c
	$(CC) -c  $< 

neutron-seq: neutron-seq.o
	$(CC)  -o $@ $+ -lm


exec: all
	./${EXE_NAME}

clean:
	rm -f ${EXE_NAME} *.o *~
