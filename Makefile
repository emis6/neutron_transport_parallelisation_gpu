SRCS = neutron-omp.cu neutron-omp2.cu neutron.cu neutron-seq.c
EXE_NAME = neutron-omp neutron-omp2 neutron neutron-seq
OBJECTS = neutron-omp.o neutron-omp2.o neutron.o neutron-seq.o

CC = gcc
CFLAGS = -O3 -arch=sm_20 -lineinfo   #-std=c11
LIB=-lm -L/usr/local/cuda/lib64/ -lcuda -lcudart #-openmp
NVCC= /usr/local/cuda/bin/nvcc -Xcompiler -fopenmp


all: ${EXE_NAME}

neutron-omp.o : neutron-omp.cu
	$(NVCC) -c $(CFLAGS) $< 

neutron-omp : neutron-omp.o
	${NVCC} ${CFLAGS} -o $@ $+ ${LIB}

neutron-omp2.o : neutron-omp2.cu
	$(NVCC) -c $(CFLAGS) $< 

neutron-omp2 : neutron-omp2.o
	${NVCC} ${CFLAGS} -o $@ $+ ${LIB}

neutron.o : neutron.cu
	$(NVCC) -c $(CFLAGS) $< 

neutron : neutron.o
	${NVCC} ${CFLAGS} -o $@ $+ ${LIB}

neutron-seq.o : neutron-seq.c
	$(CC) -c  $< 

neutron-seq: neutron-seq.o
	$(CC)  -o $@ $+ -lm




clean:
	rm -f ${EXE_NAME} *.o *~
