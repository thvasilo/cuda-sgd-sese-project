from __future__ import print_function
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
import numpy as np

Ndata = '10000'
cmd = './gradient'

times = []
Nblocks = []
batch_sizes_iter = np.arange(1, 3500, 100)
batch_sizes = []

for batch_size in batch_sizes_iter:
    # Transform to script
    batch_size = str(batch_size)
    print("Experiment for : " + batch_size)

    # Run the process
    p = Popen([cmd, batch_size, Ndata], stdout=PIPE)

    # Get the output
    output = p.stdout.read()
    lines = output.splitlines()

    # Exctract the information
    time = lines[0].split('=')
    blocks = lines[1].split('=')
    threads = lines[2].split('=')

    times.append(time[1])
    Nblocks.append(blocks[1])
    batch_sizes.append(threads[1])


# Transform the inputs from string to units
times = [float(t) for t in times]
Nblocks = [int(N) for N in Nblocks]    
batch_sizes = [int(N) for N in batch_sizes]


# Plot the data
plt.plot(batch_sizes, times, '*-')
plt.xlabel('batch_sizes')
plt.ylabel('time to execute (ms)')
plt.show()
