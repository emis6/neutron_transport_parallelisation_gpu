SRCS = neutron.cu
EXE_NAME = neutron
OBJECTS = neutron.o

CC = gcc
CFLAGS = -O3 -arch=sm_20 -lineinfo   #-std=c11
LIB=-lm -L/usr/local/cuda/lib64/ -lcuda -lcudart
NVCC= /usr/local/cuda/bin/nvcc 


all: ${EXE_NAME}

neutron.o : neutron.cu
	$(NVCC) -c $(CFLAGS) $< 

neutron : neutron.o
	${NVCC} ${CFLAGS} -o $@ $+ ${LIB}

clean:
	rm -f ${EXE_NAME} *.o *~
