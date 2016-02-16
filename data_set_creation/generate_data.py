"""
This generates the data files for testing
"""

from __future__ import print_function
import numpy as np
from sklearn.datasets import make_regression
import csv
import sys
import getopt


def create_data_set(n_samples, n_features, noise=0, filename='./test', seed=42):
    """
    Creates data from a linear model and the model. The model has n_features
    and n_samples. The level of noise can be controled with the noise comand
    Finally the data is saved in filename and the weights are saved
    as well.
    """

    # Change this for reproducibility issues
    prng = np.random.RandomState(seed)

    # Generate data
    n_informative = n_features  # All feature are informative
    n_targets = 1  # Only one target
    bias = 0  # No bias

    if noise == 0.0:
        data, labels, w = make_regression(n_samples, n_features, n_informative, n_targets,
                                          bias, coef=True, random_state=prng)
    else:
        data, labels, w = make_regression(n_samples, n_features, n_informative, n_targets,
                                          bias, noise,  coef=True, random_state=prng)

    # Concatenate data and labels
    to_save = np.zeros((n_samples, n_features + 1), np.float16)
    to_save[:, :n_features] = data
    to_save[:, -1] = labels

    # File destination
    filename_data = filename + '.csv'
    filename_weights = filename + '_weights' + '.csv'

    # Need to add as fail-safe in case w is dimension 1 
    w = np.array(w)  # Transform to array
    # Check if it is one dimensional
    if w.ndim == 0:
        w = w[..., np.newaxis]

    # Save data
    with open(filename_data, 'wb') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(to_save)

    # Save weights
    with open(filename_weights, 'wb') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(w)

    return data, labels, w


def usage():
    print(["--samples", "--features", "--noise", "--filename", "--seed"])


def main(argv):
    # Parameters
    n_samples = 10000
    n_features = 1
    noise = 0.0
    seed = 42
    filename = './python_data'

    try:
        opts, args = getopt.getopt(
            argv,
            "s:f:n:e:o:",
            ["samples", "features", "noise", "filename", "seed"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ["-s", "--samples"]:
            n_samples = int(arg)
        elif opt in ("-f", "--features"):
            n_features = int(arg)
        elif opt in ("-n", "--noise"):
            noise = float(arg)
        elif opt in ("-o", "--filename"):
            filename = arg
        elif opt in ("-e", "--seed"):
            seed = int(arg)

    create_data_set(n_samples, n_features, noise, filename, seed)

if __name__ == '__main__':
    main(sys.argv[1:])
