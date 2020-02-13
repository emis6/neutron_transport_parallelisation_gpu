SRCS =  neutron-par-cpu.cpp
EXE_NAME = neutron-par-cpu
OBJECTS = neutron-par-cpu.o

CC = gcc

all: ${EXE_NAME}

neutron-par-cpu.o : neutron-par-cpu.cpp
	g++ -c -O3  $< 

neutron-par-cpu: neutron-par-cpu.o
	g++  -o $@ $+ -lm -fopenmp


exec: all
	./${EXE_NAME}

clean:
	rm -f ${EXE_NAME} *.o *~
