SRCS =  ./gpu/neutron.cu ./seq/neutron-seq.c ./hybrid/neutron-test.cu ./cpu_parallel/neutron-par-cpu.cpp
EXE_NAME = ./gpu/neutron ./seq/neutron-seq ./hybrid/neutron-test ./cpu_parallel/neutron-par-cpu
OBJECTS = ./gpu/neutron.o ./seq/neutron-seq.o ./hybrid/neutron-test.o ./cpu_parallel/neutron-par-cpu.o

CC = gcc
CFLAGS = -O3 -arch=sm_20 -lineinfo   #-std=c11
LIB=-lm -L/usr/local/cuda/lib64/ -lcuda -lcudart #-openmp
NVCC= /usr/local/cuda/bin/nvcc -Xcompiler -fopenmp


all: ${EXE_NAME}


./gpu/neutron.o : ./gpu/neutron.cu
	$(NVCC) -c $(CFLAGS) $< 

./gpu/neutron : ./gpu/neutron.o
	${NVCC} ${CFLAGS} -o $@ $+ ${LIB}


./hybrid/neutron-test.o : ./hybrid/neutron-test.cu
	$(NVCC) -c $(CFLAGS) $< 

./hybrid/neutron-test : ./hybrid/neutron-test.o
	${NVCC} ${CFLAGS} -o $@ $+ ${LIB}


./seq/neutron-seq.o : neutron-seq.c
	$(CC) -c  $< 

./seq/neutron-seq: neutron-seq.o
	$(CC)  -o $@ $+ -lm


./cpu_parallel/neutron-par-cpu.o : neutron-par-cpu.cpp
	g++ -c -O3  $< 

./cpu_parallel/neutron-par-cpu: neutron-par-cpu.o
	g++  -o $@ $+ -lm -fopenmp



clean:
	rm -f ${EXE_NAME} *.o *~
