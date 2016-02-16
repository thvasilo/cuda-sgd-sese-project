from __future__ import print_function
from generate_data import create_data_set
import numpy as np


def generate_data_for_features(n_samples, feature_sizes_vector, noise, filename, seed):
    """
    This function will produce and save a set of data sets to
    test the scalability with features. In order to do so the
    function takes a vector with the number of features that
    the the user wishes to get tested. 

    So for example if the vector is (1, 5, 10) three data sets
    will be generated with the same number of data (samples), 
    the same noise but with number of features 1, 5, and 10
    respectively.
    """

    for n_features in feature_sizes_vector:
        name = filename + str(n_features)
        create_data_set(n_samples, n_features, noise, name, seed)
        

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Script to create a number of csv files for testing scaling"
                                                 "in terms of number of data points")
    parser.add_argument("--start", help="Starting point for the number of samples", type=int,
                        required=True)
    parser.add_argument("--end", help="Ending point (inclusive) for the number of samples", type=int,
                        required=True)
    parser.add_argument("--stride", help="Stride for the number of samples", type=int,
                        required=True)
    parser.add_argument("--samples", help="Number of features the dataset should contain", type=int,
                        required=True)
    parser.add_argument("--seed", help="Seed to easy replication of data generation", type=int,
                        default=0)
    parser.add_argument("--noise", help="Amount of noise to add to the generated data", type=float,
                        default=0.0)

    args = parser.parse_args()

    # Parameters
    max_features = args.end
    n_samples = args.samples
    noise = args.noise
    seed = args.seed

    jump = args.stride
    start = args.start
    feature_sizes_vector = np.arange(start, max_features + jump, jump)
    filename = './test_number_of_features-'

    # Call the function
    generate_data_for_features(n_samples, feature_sizes_vector, noise, filename, seed)


if __name__ == '__main__':
    main()
