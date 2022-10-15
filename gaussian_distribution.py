import numpy as np
import matplotlib.pyplot as plt


def get_gaussian_class_mean(class_data):
    gaussian_mean = np.sum(class_data, axis=0)/(len(class_data))
    return gaussian_mean


def get_gaussian_covariance_matrix(class_data, class_mean):
    diffs = class_data - class_mean
    diffs = diffs[:, :, None]
    cov_matrix = np.ndarray([diffs.shape[0], diffs.shape[1], diffs.shape[1]])
    for index in range(diffs.shape[0]):
        cov_matrix[index, :, :] = (diffs[index, :]) @ diffs[index, :].T
    covariance_transpose = np.sum(cov_matrix, 0)/class_data.shape[0]
    covariance = covariance_transpose.T
    return covariance


def gaussian_probability_distribution(mean, covariance, test_vector):
    n = mean.shape[0]
    sigma_determinant = np.linalg.det(covariance)
    sigma_inverse = np.linalg.inv(covariance)
    diff = test_vector - mean
    dist = (diff.T @ sigma_inverse @ diff)
    exponent = (-1/2)*dist
    gaussian_dist = ((2*np.pi)**(-n/2)) * (sigma_determinant**(-1/2)) * np.exp(exponent)
    return gaussian_dist


def decision_boundary(class_zero_data, class_one_data):
    plt.scatter(class_zero_data[:, 0], class_zero_data[:, 1])
    plt.scatter(class_one_data[:, 0], class_one_data[:, 1])

    plt.show()
