from __future__ import print_function
from subprocess import Popen, PIPE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import json
import os


if __name__ == '__main__':
    """
    Here we will plot how the system scales with respect to the number of samples.
    In order to run this script the data has to be generated in data set
    creation. To do this use the generate_scalable_data.py script
    """

    # Command to run.
    # We are assuming execution from <repo-root>/experiments/n_samples
    cmd = '../../c_code/main'
    # Set run parameters
    # TODO: Take at least some of these parameters as args
    learning_rate = 0.01
    iterations = 10
    base_path = '../data/n_samples/'  # TODO: data dir should be under repo root
    base_name = 'test_number_of_samples'
    full_path = base_path + base_name
    weights_file = full_path + '_weights' + '.csv'
    batch_size = 100
    R = 1000
    C = 3

    # Read the weights
    with open(weights_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            true_weights = row

    true_weights = np.asarray([float(w) for w in true_weights])

    # Initialize lists to store
    memory_times = []
    times = []
    errors = []
    weights_collection = []
    kernel_times = []

    # TODO: These _need_ to be set from parameters
    n_samples = 10000
    jump = 500
    start = 1000
    data_subsets_sizes = np.arange(start, n_samples + jump, jump)

    # Now we run the process
    for R in data_subsets_sizes:
        print('Data size', R)
        filename = base_name + str(R)
        dataset = full_path + str(R) + '.csv'
        print('data set', dataset)

        parameters = [learning_rate, iterations, dataset, R, C, batch_size]
        parameters = [str(par) for par in parameters]
        run_line = [cmd] + parameters
        # Run the command with the main CUDA program
        print("Running cmd: {}".format(run_line))
        p = Popen(run_line)
        p.wait()

        # Read JSON experiment output file
        experiment_vals = {}
        with open(filename + "-output.json", 'r') as f:
            json_string = f.read()
            experiment_vals = json.loads(json_string)

        # Extract information
        memory_times.append(experiment_vals["transfer_time"])

        # error = lines[1].split(':')
        # errors.append(error[1])
        #
        # weights = lines[2].split('=')[1]
        # weights = weights[2:-2]
        # weights = weights.split(' ')[1:]
        # weights = [float(w) for w in weights]
        # weights_collection.append(np.asarray(weights))

        kernel_times.append(experiment_vals["gpu_time"])

    # Transform the relevant quantities to lists
    # kernel_times = [float(t) for t in kernel_times]
    # memory_times = [float(t) for t in memory_times]
    # errors = np.asarray([float(e) for e in errors])
    # weights_errors = [np.linalg.norm(true_weights - ws) for ws in weights_collection]

    # Plotting

    # Configuration
    if not os.path.exists("plots"):
        os.makedirs("plots")
    folder = './plots/'
    format_to_save = '.pdf'
    frameon = False  # no background
    transparent = True
    dpi = 1200
    fontsize = 18


    def create_plot(filename, title, x_label, y_label):
        # First the kernel time
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data_subsets_sizes, kernel_times, '-o')
        ax.set_xlim(0, 10100)
        ax.set_ylim(bottom=0)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Change the font size
        axes = fig.get_axes()
        for ax in axes:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)

        filename_to_save = folder + filename + format_to_save
        fig.tight_layout()
        fig.savefig(filename_to_save, frameon=frameon,
                    transparent=transparent, dpi=dpi, bbox_indces='tight')
        plt.close(fig)

    # First the kernel time
    create_plot(
            filename='kernel_time_n_samples',
            title='Kernel time scaling with the number of samples',
            x_label='Num. samples',
            y_label='GPU Time')

    create_plot(
            filename='transfer_time_n_samples',
            title='Data transfer time scaling with the number of samples',
            x_label='Num. samples',
            y_label='Data transfer Time')

# TODO: Decide what else I want to be plotting
# Now the total error
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(data_subsets_sizes, errors, '-o')
# ax.set_xlim(0, 10100)
# ax.set_ylim(bottom=0)
#
# ax.set_title('Error scaling with number of samples')
# ax.set_xlabel('N samples')
# ax.set_ylabel('Sum of Squares Errors (SSE)')
#
# # Change the font size
# axes = fig.get_axes()
# for ax in axes:
#     for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#                  ax.get_xticklabels() + ax.get_yticklabels()):
#         item.set_fontsize(fontsize)
#
#
# name = 'errors_n_samples'
# filename_to_save =  folder + name + format_to_save
# fig.tight_layout()
# fig.savefig(filename_to_save, frameon=frameon,
#             transparent=transparent, dpi=dpi, bbox_indces='tight')
# plt.close(fig)
#
# # Finally the weights errors
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(data_subsets_sizes, weights_errors, '-o')
# ax.set_xlim(0, 10100)
# ax.set_ylim(bottom=0, top=2 * np.max(weights_errors))
#
# ax.set_title('Weights prediction error')
# ax.set_xlabel('N samples')
# ax.set_ylabel('Weight Error')
#
# # Change the font size
# axes = fig.get_axes()
# for ax in axes:
#     for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#                  ax.get_xticklabels() + ax.get_yticklabels()):
#         item.set_fontsize(fontsize)
#
# name = 'weights_errors_n_samples'
# filename_to_save =  folder + name + format_to_save
#
# fig.tight_layout()
# fig.savefig(filename_to_save, frameon=frameon,
#             transparent=transparent, dpi=dpi, bbox_indces='tight')
# plt.close(fig)
