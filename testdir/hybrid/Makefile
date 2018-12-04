SRCS =  neutron-test.cu 
EXE_NAME = neutron-test  
OBJECTS = neutron-test.o 

CC = gcc
CFLAGS = -O3 -arch=sm_20 -lineinfo   #-std=c11
LIB=-lm -L/usr/local/cuda/lib64/ -lcuda -lcudart #-openmp
NVCC= /usr/local/cuda/bin/nvcc -Xcompiler -fopenmp


all: ${EXE_NAME}


neutron-test.o : neutron-test.cu
	$(NVCC) -c $(CFLAGS) $< 

neutron-test : neutron-test.o
	${NVCC} ${CFLAGS} -o $@ $+ ${LIB}


clean:
	rm -f ${EXE_NAME} *.o *~
