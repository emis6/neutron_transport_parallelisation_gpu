import sys
import os
from subprocess import PIPE, Popen


def getTotTime(out):
	print("this is out:-----\n")
	print(out)
	print("\n")

	array_de_lines = out.split("\n");
	print("------------->" + array_de_lines[7])
	temps = array_de_lines[7].split(":")[1];
	temps = temps.split(" ")[1];
	print("------------->" + temps)

	return float(temps);


NN = 10
h = 1
n = [100000000,200000000,500000000, 900000000]

sum_time_seq = 0;
sum_time_omp = 0;
sum_time_gpu = 0;
sum_time_hybrid = 0;

for nombre in n:
	p = Popen(['./neutron-omp'], shell=True, stdin=PIPE, stdout=PIPE)
	out, err = p.communicate()
	sum_time_seq += getTotTime(out.decode("utf-8"))

print(sum_time_seq + " + " + sum_time_seq/4)