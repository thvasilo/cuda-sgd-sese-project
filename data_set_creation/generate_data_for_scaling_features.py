"""
This script generates the data sets for tests scalability
of the features. 
"""

from __future__ import print_function
from generate_data import create_data_set
import numpy as np

def generate_data_for_features(n_samples, vector_of_number_of_features, noise=0, 
                               filename='./test', seed=42):
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

    for n_features in vector_of_number_of_features:
        name = filename + str(n_features)
        create_data_set(n_samples, n_features, noise, name, seed)
        

def main():
    n_samples = 10000
    noise = 0.0
    filename = './features_scalability'
    feature_vector = np.arange(2, 16, 1)

    # Call the function
    generate_data_for_features(n_samples, feature_vector, noise, filename)


if __name__ == '__main__':
    main()
