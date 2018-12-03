SRCS = neutron-omp.cu neutron-omp2.cu neutron.cu neutron-seq.c neutron-test.cu neutron-par-cpu.cpp
EXE_NAME = neutron-omp neutron-omp2 neutron neutron-seq neutron-test neutron-par-cpu
OBJECTS = neutron-omp.o neutron-omp2.o neutron.o neutron-seq.o neutron-test.o neutron-par-cpu.o

CC = gcc
CFLAGS = -O3 -arch=sm_20 -lineinfo   #-std=c11
LIB=-lm -L/usr/local/cuda/lib64/ -lcuda -lcudart #-openmp
NVCC= /usr/local/cuda/bin/nvcc -Xcompiler -fopenmp


all: ${EXE_NAME}

neutron.o : neutron.cu
	$(NVCC) -c $(CFLAGS) $< 

neutron : neutron.o
	${NVCC} ${CFLAGS} -o $@ $+ ${LIB}


neutron-test.o : neutron-test.cu
	$(NVCC) -c $(CFLAGS) $< 

neutron-test : neutron-test.o
	${NVCC} ${CFLAGS} -o $@ $+ ${LIB}


neutron-par-cpu.o : neutron-par-cpu.cpp
	g++ -c -O3  $< 

neutron-par-cpu: neutron-par-cpu.o
	g++  -o $@ $+ -lm -fopenmp


neutron-seq.o : neutron-seq.c
	$(CC) -c  $< 

neutron-seq: neutron-seq.o
	$(CC)  -o $@ $+ -lm




clean:
	rm -f ${EXE_NAME} *.o *~
