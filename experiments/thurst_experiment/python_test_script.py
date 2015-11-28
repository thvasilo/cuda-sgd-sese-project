from __future__ import print_function
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
import numpy as np


# Set parameters
cmd = './main'

learning_rate = 0.5
iterations = 200
dataset = './data/test.csv'
batch_size = 100
R = 768
C = 7

# Now we run the process
parameters = [learning_rate, iterations, dataset, R, C, batch_size]
parameters = [str(par) for par in parameters]
run_line = [cmd]
run_line.extend(parameters)
p = Popen(run_line, stdout=PIPE)
output = p.stdout.read()
