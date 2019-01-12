SRCS =  neutron-seq.c 
EXE_NAME =neutron-seq
OBJECTS = neutron-seq.o

CC = gcc

all: ${EXE_NAME}

neutron-seq.o : neutron-seq.c
	$(CC) -c  $< 

neutron-seq: neutron-seq.o
	$(CC)  -o $@ $+ -lm


exec: all
	./${EXE_NAME}

clean:
	rm -f ${EXE_NAME} *.o *~
