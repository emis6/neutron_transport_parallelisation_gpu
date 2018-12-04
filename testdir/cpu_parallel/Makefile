SRCS =  neutron-par-cpu.cpp
EXE_NAME = neutron-par-cpu
OBJECTS = neutron-par-cpu.o

CC = gcc
CFLAGS = -O3 -arch=sm_20 -lineinfo   #-std=c11
LIB=-lm -L/usr/local/cuda/lib64/ -lcuda -lcudart #-openmp
NVCC= /usr/local/cuda/bin/nvcc -Xcompiler -fopenmp


all: ${EXE_NAME}

neutron-par-cpu.o : neutron-par-cpu.cpp
	g++ -c -O3  $< 

neutron-par-cpu: neutron-par-cpu.o
	g++  -o $@ $+ -lm -fopenmp


exec: all
	./${EXE_NAME}

clean:
	rm -f ${EXE_NAME} *.o *~
