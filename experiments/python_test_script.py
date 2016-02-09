from __future__ import print_function
from subprocess import Popen, PIPE
import csv
import matplotlib.pyplot as plt
import numpy as np

# Set parameters
cmd = './main'

learning_rate = 0.5
iterations = 1000
name = './data/python_data'
dataset = name + '.csv'
weights_file = name + '_weights' + '.csv'
batch_size = 100
R = 1000
C = 3

# Read the weights
with open(weights_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        true_weights = row

# Now we run the process
parameters = [learning_rate, iterations, dataset, R, C, batch_size]
parameters = [str(par) for par in parameters]
run_line = [cmd]
run_line.extend(parameters)
p = Popen(run_line, stdout=PIPE)

# Get output per line
output = p.stdout.read()
lines = output.splitlines()

# Extract information
memory_time = lines[0].split('=')
error = lines[1].split(':')
weights = lines[2].split('=')[1]
weights = weights[2:-2]
weights = weights.split(' ')[1:]
kernel_time = lines[3].split('=')
