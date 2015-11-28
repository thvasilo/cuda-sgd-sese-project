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

# Iterate over batch_sizes
batch_sizes = np.arange(10, 100, 10)
times = []

# Now we run the process
for batch_size in batch_sizes:
    parameters = [learning_rate, iterations, dataset, R, C, batch_size]
    parameters = [str(par) for par in parameters]
    run_line = [cmd]
    run_line.extend(parameters)
    p = Popen(run_line, stdout=PIPE)

    # Get output per line
    output = p.stdout.read()
    lines = output.splitlines()

   # Extract information
    time = lines[0].split('=')
    times.append(time[1])

# Plot everything
times = [float(t) for t in times]
plt.plot(batch_sizes, times)
plt.show()

