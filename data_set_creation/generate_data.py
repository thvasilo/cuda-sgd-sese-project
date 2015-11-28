"""
This generates the data files for testing
"""

from __future__ import print_function
import numpy as np
from sklearn.datasets import make_regression
import csv

def create_data_set(n_samples, n_features, noise=0, filename='./test'):
    """
    Creates data from a linear model and the model. The model has n_features
    and n_samples. The level of noise can be controled with the noise comand
    Finally the data is saved in filename. Furthermore the weights are saved
    as well.
    """
    
    # Change this for reproducibility issues
    seed = 10
    prng = np.random.RandomState(seed)

    # Generate data
    n_informative = n_features  # All feature are informative
    n_targets = 1  # Only one target 
    bias = 0  # No bias
    
    data, labels, w = make_regression(n_samples, n_features, n_informative, n_targets,
                                      bias, noise,  coef=True, random_state=prng)
    # Concatenate data and labels
    to_save = np.zeros((n_samples, n_features + 1), np.float16)
    to_save[:, :n_features] = data
    to_save[:, -1] = labels

    # File destination
    filename_data = filename + '.csv'
    filename_weights = filename + '_weights' + '.csv'

    # Save data
    with open(filename_data, 'wb') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(to_save)

    # Save weights
    with open(filename_weights, 'wb') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerow(w)

    return data, labels, w
        
# Parameters
n_samples = 10
n_features = 3
noise = 1.0
filename = './test'
# Run the program
X, y, w = create_data_set(n_samples, n_features, noise, filename)
