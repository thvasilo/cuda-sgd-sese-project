from __future__ import print_function
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
import numpy as np
import csv


# Set parameters
cmd = './main'

learning_rate = 0.001
iterations = 1000
name = './data/correct_prediction'
dataset = name + '.csv'
weights_file = name + '_weights' + '.csv'
batch_size = 100
R = 1000
C = 3

# Read the weightsbat
with open(weights_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        true_weights = row

true_weights = [float(w) for w in true_weights]
true_weights = np.asarray(true_weights)
        
# Iterate over batch_sizes
batch_sizes = np.arange(10, 100, 10)

# Initialize list to store
memory_times = []
times = []
errors = []
weights_collection = []
kernel_times = []

# Now we run the process
for batch_size in batch_sizes:
    parameters = [learning_rate, iterations, dataset, R, C, batch_size]
    parameters = [str(par) for par in parameters]
    run_line = [cmd] + parameters
    p = Popen(run_line, stdout=PIPE)

    # Get output per line
    output = p.stdout.read()
    lines = output.splitlines()

    # Extract information
    memory_time = lines[0].split('=')
    memory_times.append(memory_time[1])

    error = lines[1].split(':')
    errors.append(error[1])

    weights = lines[2].split('=')[1]
    weights = weights[2:-2]
    weights = weights.split(' ')[1:]
    weights = [float(w) for w in weights]
    weights_collection.append(np.asarray(weights))

    kernel_time = lines[3].split('=')
    kernel_times.append(kernel_time[1])


# Transform to arrays the relevant quantities
kernel_times = [float(t) for t in kernel_times]
errors = np.asarray([float(e) for e in errors])
weights_errors = [np.linalg.norm(true_weights - ws) for ws in weights_collection]

# Plot everything
# plt.plot(batch_sizes, kernel_times)
# plt.show()

# Now plot everything
plt.subplot(2, 1, 1)
plt.title('weight_errors')
plt.plot(batch_sizes, weights_errors, '*-')
plt.subplot(2, 1, 2)
plt.title('errors')
plt.plot(batch_sizes, errors, '*-')
plt.show()

