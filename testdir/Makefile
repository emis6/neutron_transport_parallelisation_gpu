SRCS =  ./gpu/neutron.cu ./seq/neutron-seq.c ./hybrid/neutron-hyb.cu ./cpu_parallel/neutron-par-cpu.cpp
EXE_NAME = neutron neutron-seq neutron-hyb  neutron-par-cpu
OBJECTS = neutron.o neutron-seq.o neutron-hyb.o neutron-par-cpu.o

CC = gcc
CFLAGS = -O3 -arch=sm_20 -lineinfo   #-std=c11
LIB=-lm -L/usr/local/cuda/lib64/ -lcuda -lcudart #-openmp
NVCC= /usr/local/cuda/bin/nvcc -Xcompiler -fopenmp



all: ${EXE_NAME}


neutron.o : ./gpu/neutron.cu
	$(NVCC) -c $(CFLAGS) $< 

neutron : neutron.o
	${NVCC} ${CFLAGS} -o $@ $+ ${LIB}


neutron-hyb.o : ./hybrid/neutron-hyb.cu
	$(NVCC) -c $(CFLAGS) $< 

neutron-hyb : neutron-hyb.o
	${NVCC} ${CFLAGS} -o $@ $+ ${LIB}


neutron-seq.o : ./seq/neutron-seq.c
	$(CC) -c  $< 

neutron-seq: neutron-seq.o
	$(CC)  -o $@ $+ -lm


neutron-par-cpu.o : ./cpu_parallel/neutron-par-cpu.cpp
	g++ -c -O3  $< 

neutron-par-cpu: neutron-par-cpu.o
	g++  -o $@ $+ -lm -fopenmp



clean:
	rm -f ${EXE_NAME} *.o *~
