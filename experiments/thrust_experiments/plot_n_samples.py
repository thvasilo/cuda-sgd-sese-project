"""
Here we will plot how the system scales with respect to the number of samples.
In order to run this script the data has to be generated in data set 
creation. To do this use the generate_scalable_data.py script
"""


from __future__ import print_function
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
import numpy as np
import csv
import seaborn as sns


# Set parameters
cmd = './main'  # The name of the program

learning_rate = 0.01
iterations = 1000
base_name = './data/n_samples/test_number_of_samples'
weights_file = base_name + '_weights' + '.csv'
batch_size = 100
R = 1000
C = 3

# Read the weights 
with open(weights_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        true_weights = row

true_weights = [float(w) for w in true_weights]
true_weights = np.asarray(true_weights)
        
# Initialize lists to store
memory_times = []
times = []
errors = []
weights_collection = []
kernel_times = []

n_samples = 10000
jump = 500
start = 1000
data_subsets_sizes = np.arange(start, n_samples + jump, jump)

# Now we run the process
for R in data_subsets_sizes:
    print('Data size', R)
    name = base_name + str(R)
    dataset = name + '.csv'
    print('data set', dataset)
    
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
memory_times = [float(t) for t in memory_times]
errors = np.asarray([float(e) for e in errors])
weights_errors = [np.linalg.norm(true_weights - ws) for ws in weights_collection]

## Plotting
# Configuration
folder = './'
format_to_save = '.pdf'
frameon = False  # no background
transparent = True
dpi = 1200
fontsize = 18

# First the kernel time
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data_subsets_sizes, kernel_times, '-o')
ax.set_xlim(0, 10100)
ax.set_ylim(bottom=0)

ax.set_title('Kernel time scaling with the number of samples')
ax.set_xlabel('N samples')
ax.set_ylabel('Kernel time (ms)')

# Change the font size
axes = fig.get_axes()
for ax in axes:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

name = 'kernel_time_n_samples'
filename_to_save =  folder + name + format_to_save
fig.tight_layout()
fig.savefig(filename_to_save, frameon=frameon,
            transparent=transparent, dpi=dpi, bbox_indces='tight')
plt.close(fig)

# Second the memory time
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data_subsets_sizes, memory_times, '-o')
ax.set_xlim(0, 10100)
ax.set_ylim(bottom=0, top=25.0)

ax.set_title('Memory time scaling')
ax.set_xlabel('N samples')
ax.set_ylabel('Memory time (ms)')

# Change the font size
axes = fig.get_axes()
for ax in axes:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

name = 'memory_time_n_samples'
filename_to_save =  folder + name + format_to_save
fig.tight_layout()
fig.savefig(filename_to_save, frameon=frameon,
            transparent=transparent, dpi=dpi, bbox_indces='tight')
plt.close(fig)

# Now the total error
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data_subsets_sizes, errors, '-o')
ax.set_xlim(0, 10100)
ax.set_ylim(bottom=0)

ax.set_title('Error scaling with number of samples')
ax.set_xlabel('N samples')
ax.set_ylabel('Sum of Squares Errors (SSE)')

# Change the font size
axes = fig.get_axes()
for ax in axes:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)


name = 'errors_n_samples'
filename_to_save =  folder + name + format_to_save
fig.tight_layout()
fig.savefig(filename_to_save, frameon=frameon,
            transparent=transparent, dpi=dpi, bbox_indces='tight')
plt.close(fig)

# Finally the weights errors
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data_subsets_sizes, weights_errors, '-o')
ax.set_xlim(0, 10100)
ax.set_ylim(bottom=0, top=2 * np.max(weights_errors))

ax.set_title('Weights prediction error')
ax.set_xlabel('N samples')
ax.set_ylabel('Weight Error')

# Change the font size
axes = fig.get_axes()
for ax in axes:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

name = 'weights_errors_n_samples'
filename_to_save =  folder + name + format_to_save

fig.tight_layout()
fig.savefig(filename_to_save, frameon=frameon,
            transparent=transparent, dpi=dpi, bbox_indces='tight')
plt.close(fig)
