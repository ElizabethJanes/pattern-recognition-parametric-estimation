import numpy as np


def get_class_lambda(class_data):
    x_sum = np.sum(class_data, axis=0)
    class_lambda = len(class_data)/x_sum
    return class_lambda


def exponential_probability_distribution(test_point, mean=None, covariance=None, class_lambda=None, a=None, b=None):
    if test_point >= 0:
        probability = class_lambda * np.exp(-class_lambda * test_point)
    else:
        probability = 0
    return probability
