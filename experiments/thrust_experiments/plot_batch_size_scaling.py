###
# Here we will plot how the kernel and memory time for different batch sizes
# and different data set sizes. 
###

from __future__ import print_function
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
import numpy as np
import csv
import seaborn as sns


# Set parameters
cmd = './main'

learning_rate = 0.01
iterations = 1000
base_name = './data/scalable_batch_size'
dataset = base_name + '.csv'
weights_file = base_name + '_weights' + '.csv'
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
batch_sizes = np.arange(10, 100, 5)

## Plotting
# Configuration
folder = './'
format_to_save = '.pdf'
frameon = False  # no background
transparent = True
dpi = 1200
fontsize = 18

# We define the figures
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

n_samples = 3000
jump = 1000
start = 1000
data_subsets_sizes = np.arange(start, n_samples + jump, jump)

for R in data_subsets_sizes:
    name = base_name + str(R)
    print(name)

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

    # Add the plot to the axis
    label = 'Ns = ' + str(R)
    ax1.plot(batch_sizes, kernel_times, '-o', label=label)
    ax2.plot(batch_sizes, memory_times, '-o', label=label)

# Now customize the plot 1 kernel times
ax1.set_xlim(0, 100)
ax1.set_ylim(bottom=0)
ax1.legend()

ax1.set_title('Kernel time scaling')
ax1.set_xlabel('Batch size')
ax1.set_ylabel('Kernel time (ms)')

# Change the font size
axes = fig1.get_axes()
for ax in axes:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

name = 'batch_size_scaling_kernel'
filename_to_save =  folder + name + format_to_save
fig1.tight_layout()
fig1.savefig(filename_to_save, frameon=frameon,
            transparent=transparent, dpi=dpi, bbox_indces='tight')
plt.close(fig1)

# Now customize the plot2 memory times
ax2.set_xlim(0, 100)
ax2.set_ylim(bottom=0, top=4.0)
ax2.legend()

ax2.set_title('Memory time scaling')
ax2.set_xlabel('Batch size')
ax2.set_ylabel('Memory time (ms)')

# Change the font size
axes = fig2.get_axes()
for ax in axes:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

name = 'batch_size_scaling_memory'
filename_to_save =  folder + name + format_to_save
fig2.tight_layout()
fig2.savefig(filename_to_save, frameon=frameon,
            transparent=transparent, dpi=dpi, bbox_indces='tight')
plt.close(fig2)
