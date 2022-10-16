import numpy as np


def get_class_parameters(class_data):
    a = np.amin(class_data)
    b = np.amax(class_data)
    return a, b


def uniform_probability_distribution(test_point, mean=None, covariance=None, class_lambda=None, a=None, b=None):
    if a <= test_point <= b:
        probability = 1/(b-a)
    else:
        probability = 0
    return probability
