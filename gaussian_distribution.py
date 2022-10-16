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


def gaussian_probability_distribution(test_vector, mean, covariance, class_lambda=None, a=None, b=None):
    n = mean.shape[0]
    sigma_determinant = np.linalg.det(covariance)
    sigma_inverse = np.linalg.inv(covariance)
    diff = test_vector - mean
    dist = (diff.T @ sigma_inverse @ diff)
    exponent = (-1/2)*dist
    gaussian_dist = ((2*np.pi)**(-n/2)) * (sigma_determinant**(-1/2)) * np.exp(exponent)
    return gaussian_dist


def gaussian_ml_decision_boundary(c_zero_mean, c_zero_covariance, c_one_mean, c_one_covariance, x):
    c_zero_covariance_inverse = np.linalg.inv(c_zero_covariance)
    c_zero_covariance_determinant = np.linalg.det(c_zero_covariance)
    c_one_covariance_inverse = np.linalg.inv(c_one_covariance)
    c_one_covariance_determinant = np.linalg.det(c_one_covariance)

    q0 = c_zero_covariance_inverse - c_one_covariance_inverse
    q1 = 2 * (np.dot(c_one_mean, c_one_covariance_inverse) - np.dot(c_zero_mean, c_zero_covariance_inverse))
    q2 = (np.dot(np.dot(c_zero_mean, c_zero_covariance_inverse), c_zero_mean)) - (np.dot(np.dot(c_one_mean, c_one_covariance_inverse), c_one_mean))
    q3 = np.log(c_zero_covariance_determinant) - np.log(c_one_covariance_determinant)

    decision_boundary_point = np.dot(np.dot(x, q0), x) + np.dot(q1, x) + q2 + q3
    return decision_boundary_point


def decision_boundary(class_zero_data, class_one_data, c_zero_mean, c_zero_covariance, c_one_mean, c_one_covariance):
    step = 5
    x1 = np.arange(-1500, 2500, step)
    x2 = np.arange(-1000, 2000, step)

    function_values = np.ndarray((x2.shape[0], x1.shape[0]), dtype=np.float32)

    for i in range(x2.shape[0]):
        for j in range(x1.shape[0]):
            function_val = gaussian_ml_decision_boundary(
                c_zero_mean, c_zero_covariance, c_one_mean, c_one_covariance, np.array([x1[j], x2[i]])
            )
            function_values[i, j] = function_val

    ha, = plt.plot(class_zero_data[:, 0], class_zero_data[:, 1], 'r.', label='Class Zero')
    hb, = plt.plot(class_one_data[:, 0], class_one_data[:, 1], 'b.', label='Class One')
    ctr = plt.contour(x1, x2, function_values, levels=(0,), colors='k')
    h_bnd, _ = ctr.legend_elements()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend([ha, hb, h_bnd[0]], ['Class Zero', 'Class One', 'Classifier Boundary'])
    plt.title(f'Maximum Likelihood Classifier Decision Boundary with Gaussian Probability Distribution')

    plt.show()
