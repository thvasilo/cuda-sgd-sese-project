"""
This should generate one big data set. The program should save
smaller pieces of the initial data set as separated data sets
to test scalability.
"""

from __future__ import print_function
import numpy as np
from sklearn.datasets import make_regression
import csv


def create_scalable_data_sets(n_samples, n_features, data_subsets_sizes,
                              noise=0, filename='./test', seed=42):
    """
    This is a generalization of the create data function. This first creates
    a big data set of n_samples and then takes samller samples from it of the
    sizes provided in the data_subset_sizes vector.
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

    for data_size in data_subsets_sizes:
        # Concatenate data and labels
        to_save = np.zeros((data_size, n_features + 1))
        to_save[:, :n_features] = data[:data_size]
        to_save[:, -1] = labels[:data_size]

        # File destination
        filename_data = filename + str(data_size) + '.csv'

        # Save data
        with open(filename_data, 'wb') as fp:
            a = csv.writer(fp, delimiter=',')
            a.writerows(to_save)

    # Save weights
    filename_weights = filename + '_weights' + '.csv'
    with open(filename_weights, 'wb') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(w)

    return data, labels, w

def main():
    
    # Parameters
    n_samples = 3000
    n_features = 3
    noise = 0.0
    seed = 42

    jump = 1000
    start = 1000
    data_subsets_sizes = np.arange(start, n_samples + jump, jump)
    filename = './scalable_batch_size'

    create_scalable_data_sets(n_samples, n_features,
                              data_subsets_sizes, noise, filename, seed)

if __name__ == '__main__':
    main()
