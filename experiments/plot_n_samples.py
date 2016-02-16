from __future__ import print_function
from subprocess import Popen, PIPE
import matplotlib
import numpy as np
import json
import os
import sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    """
    This script runs a data size scaling experiment, investigating the scaling characteristics
    of the algorithm, as we increase the size of the input.
    In order to run this script the data has to be generated in data set
    creation. To do this use the generate_scalable_data.py script

    Example run from <repo-root>/experiments/experiment_folder
    > python3 ../plot_n_samples.py -i ../../data/datadir/ -p test_number_of_samples_large --start 500000 --end 1000000 \
    --stride 100000 -c 4 -b 20000 --cuda 1
    """

    import argparse
    parser = argparse.ArgumentParser(description="Script for running multiple experiments from a folder containing"
                                                 "multiple data files")
    parser.add_argument("-i", "--input", help="Directory under which the data files lie (with trailing \'/\')",
                        required=True)
    parser.add_argument("-p", "--prefix", help="Prefix name of the data files", required=True)
    parser.add_argument("--start", help="Starting point for the parameter that files vary in", type=int,
                        required=True)
    parser.add_argument("--end", help="Ending point (inclusive) for the parameter that files vary in", type=int,
                        required=True)
    parser.add_argument("--stride", help="Stride for the parameter", type=int,
                        required=True)
    parser.add_argument("--cuda", help="Set to 1 to perform a CUDA run instead of cuBLAS", default=0, type=int)
    parser.add_argument("-c", "--columns", help="Number of features in the data, if not provided we assume this is a "
                                                "n_samples experiment.", type=int, default=-1)
    parser.add_argument("-r", "--rows", help="Number of rows in the data, if not provided we assume this is a"
                                             " n_features experiment", type=int, default=-1)
    parser.add_argument("-b", "--batchsize", help="Size of mini-batch", type=int, required=True)
    parser.add_argument("-l", "--learningrate", help="Learning rate", type=float, default=0.01)
    parser.add_argument("-e", "--epochs", help="Number of epochs to run", type=int, default=10)
    # We are assuming execution from <repo-root>/experiments/experiment_folder
    parser.add_argument("--cmd", help="Command to run", type=str, default='../../c_code/main')
    parser.add_argument("--paramName", help="Name of the parameter we are investigating", type=str, required=True,
                        choices=["features", "samples"])

    args = parser.parse_args()

    # Command to run.
    cmd = args.cmd
    # Set run parameters
    learning_rate = args.learningrate
    iterations = args.epochs
    base_path = args.input  # '../data/n_samples/'  # TODO: data dir should be under repo root
    base_name = args.prefix  # 'test_number_of_samples'
    full_path = base_path + base_name
    batch_size = args.batchsize
    cuda_run = args.cuda  # When set to 1 to perform a run with the CUDA codepath instead of the cuBLAS

    # Initialize lists to store
    memory_times = []
    times = []
    errors = []
    weights_collection = []
    kernel_times = []

    # Range of values over which we iterate TODO: These could also be extracted from the filenames
    start = args.start
    n_samples = args.end
    jump = args.stride
    data_subsets_sizes = np.arange(start, n_samples + jump, jump)

    if args.columns == -1 and args.paramName == "samples":
        print("Error: Need to provide --columns argument when --paramName is set to \'samples\'")
        sys.exit(1)
    if args.rows == -1 and args.paramName == "features":
        print("Error: Need to provide --rows argument when --paramName is set to \'features\'")
        sys.exit(1)

    # Run the process for each parameter setting
    for parameter in data_subsets_sizes:
        R = args.rows if args.paramName == "features" else parameter
        C = args.columns if args.paramName == "samples" else parameter
        print('Number of data points: ', R)
        print('Number of features: ', C)
        json_filename = base_name + str(parameter)
        dataset = full_path + str(parameter) + '.csv'
        print('Data set filepath: ', dataset)

        run_parameters = [learning_rate, iterations, dataset, R, C, batch_size, cuda_run]
        run_parameters = [str(par) for par in run_parameters]
        run_line = [cmd] + run_parameters
        # Run the command with the main CUDA program
        print("Running cmd: {}".format(run_line))
        p = Popen(run_line)
        p.wait()

        # Read JSON experiment output file
        experiment_vals = {}
        suffix = "cublas" if cuda_run == 0 else "cuda"
        with open(json_filename + "-{}.json".format(suffix), 'r') as f:
            json_string = f.read()
            experiment_vals = json.loads(json_string)

        # Extract information
        memory_times.append(experiment_vals["transfer_time"])
        kernel_times.append(experiment_vals["gpu_time"])
        errors.append(experiment_vals["error"])

    # Plotting

    # Plotting configuration
    if not os.path.exists("plots"):
        os.makedirs("plots")
    folder = './plots/'
    format_to_save = '.pdf'
    frameon = False  # no background
    transparent = True
    dpi = 1200
    fontsize = 18


    def create_plot(x_data, y_data, filename, title, x_label, y_label):
        # First the kernel time
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_data, y_data, '-o')

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        filename_to_save = folder + filename + format_to_save
        fig.tight_layout()
        fig.savefig(filename_to_save, frameon=frameon,
                    transparent=transparent, dpi=dpi, bbox_indces='tight')
        plt.close(fig)

    # Plot GPU times
    create_plot(
            data_subsets_sizes,
            kernel_times,
            filename='gpu_time_n_{}'.format(args.paramName),
            title='GPU running time scaling with the number of {}'.format(args.paramName),
            x_label='Num. {}'.format(args.paramName),
            y_label='GPU Time (ms)')

    # Plot transfer times
    create_plot(
            data_subsets_sizes,
            memory_times,
            filename='transfer_time_n_{}'.format(args.paramName),
            title='Data read + transfer time scaling with the number of {}'.format(args.paramName),
            x_label='Num. {}'.format(args.paramName),
            y_label='Data read + transfer time (ms)')

    # Plot errors
    create_plot(
            data_subsets_sizes,
            errors,
            filename='error_n_{}'.format(args.paramName),
            title='Mean absolute error scaling with the number of {}'.format(args.paramName),
            x_label='Num. {}'.format(args.paramName),
            y_label='Mean Absolute Error')
