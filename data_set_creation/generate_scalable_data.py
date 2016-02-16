from __future__ import print_function
import numpy as np
from sklearn.datasets import make_regression
import csv


def create_scalable_data_sets(n_samples, n_features, data_subsets_sizes,noise, filename, seed):
    """
    This is a generalization of the create data function. This first creates
    a big data set of n_samples and then takes smaller samples from it of the
    sizes provided in the data_subset_sizes vector.
    """

    prng = np.random.RandomState(seed) if seed != 0 else np.random.RandomState()

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
        filename_data = filename + '-' + str(data_size) + '.csv'

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Script to create a number of csv files for testing scaling"
                                                 "in terms of number of data points")
    parser.add_argument("--start", help="Starting point for the number of samples", type=int,
                        required=True)
    parser.add_argument("--end", help="Ending point (inclusive) for the number of samples", type=int,
                        required=True)
    parser.add_argument("--stride", help="Stride for the number of samples", type=int,
                        required=True)
    parser.add_argument("--features", help="Number of features the dataset should contain", type=int,
                        required=True)
    parser.add_argument("--seed", help="Seed to easy replication of data generation", type=int,
                        default=0)
    parser.add_argument("--noise", help="Amount of noise to add to the generated data", type=float,
                        default=0.0)

    args = parser.parse_args()

    # Parameters
    n_samples = args.end
    n_features = args.features
    noise = args.noise
    seed = args.seed

    jump = args.stride
    start = args.start
    data_subsets_sizes = np.arange(start, n_samples + jump, jump)
    filename = './test_number_of_samples_scalable'

    create_scalable_data_sets(n_samples, n_features,
                              data_subsets_sizes, noise, filename, seed)
