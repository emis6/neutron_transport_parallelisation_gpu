import sys
import os
from subprocess import PIPE, Popen
# import matplotlib.pyplot as plt

from pylab import *

def getTotTime(out):
	temp = -1;

	array_de_lines = out.split("\n");
	
	for i in range(0,len(array_de_lines)):
		deux_arr = array_de_lines[i].split(":");
		for j in range(0, len(deux_arr)):
			if deux_arr[j]== "Temps total de calcul":
				j+= 1;
				temps = float(deux_arr[j].split(" ")[1])
				print("------------>"+ str(temps))	

	return float(temps);


NN = 10
h = 1
n = [100000000,200000000,500000000, 900000000]

time_seq = [0]*4;
time_omp = [0]*4;
time_gpu = [0]*4;
time_hyb = [0]*4;

sum_time_seq = 0;
sum_time_omp = 0;
sum_time_gpu = 0;
sum_time_hybrid = 0;

plot(n, time_gpu)
show() # affiche la figure a l'ecran
for i in range(0, len(n)):
	for blah in range(0,4):
		p = Popen(['./neutron', str(h), str(n[i]) ], shell=False, stdin=PIPE, stdout=PIPE)
		out, err = p.communicate()
		sum_time_gpu += getTotTime(out.decode("utf-8"))
	time_gpu[i] = sum_time_gpu/4.0;

	# p = Popen(['./neutron-seq', str(h), str(nombre) ], shell=False, stdin=PIPE, stdout=PIPE)
	# out, err = p.communicate()
	# sum_time_seq += getTotTime(out.decode("utf-8"))

	# p = Popen(['./neutron-omp', str(h), str(nombre) ], shell=False, stdin=PIPE, stdout=PIPE)
	# out, err = p.communicate()
	# sum_time_omp += getTotTime(out.decode("utf-8"))

	# p = Popen(['./neutron-omp2', str(h), str(nombre) ], shell=False, stdin=PIPE, stdout=PIPE)
	# out, err = p.communicate()
	# sum_time_hybrid += getTotTime(out.decode("utf-8"))


plot(n, time_gpu)

show() # affiche la figure a l'ecran
print(str(sum_time_gpu) + " + " + str(sum_time_gpu/4.0))