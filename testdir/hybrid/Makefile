SRCS =  neutron-hyb.cu 
EXE_NAME = neutron-hyb  
OBJECTS = neutron-hyb.o 

CC = gcc
CFLAGS = -O3 -arch=sm_20 -lineinfo   #-std=c11
LIB=-lm -L/usr/local/cuda/lib64/ -lcuda -lcudart #-openmp
NVCC= /usr/local/cuda/bin/nvcc -Xcompiler -fopenmp


all: ${EXE_NAME}


neutron-hyb.o : neutron-hyb.cu
	$(NVCC) -c $(CFLAGS) $< 

neutron-hyb : neutron-hyb.o
	${NVCC} ${CFLAGS} -o $@ $+ ${LIB}


exec: all
	./${EXE_NAME}
	
clean:
	rm -f ${EXE_NAME} *.o *~
