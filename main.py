import numpy as np
from dataset_utilities import get_pca_transformed_dataset, get_class_zero, get_class_one
from gaussian_distribution import get_gaussian_class_mean, get_gaussian_covariance_matrix, mathematical_decision_boundary, decision_boundary, gaussian_probability_distribution
from exponential_distribution import get_class_lambda, exponential_probability_distribution
from uniform_distribution import get_class_parameters, uniform_probability_distribution
from prediction_error import prediction_error
from maximum_likelihood_classifier import maximum_likelihood_classifier
from maximum_a_priori_classifier import get_prior_probability, maximum_a_priori_classifier


def exercise_1():
    print('Exercise 1')
    train_data_2d, train_labels_2d, test_data_2d, test_labels_2d = get_pca_transformed_dataset(2)

    train_class_zero_2d_data = get_class_zero(train_data_2d, train_labels_2d)
    train_class_one_2d_data = get_class_one(train_data_2d, train_labels_2d)

    # Question 2: Use derived expressions to find the probability distribution of the MNIST training dataset. Estimate
    # the distributions of each class by finding the mean vectors and covariance matrices of each class.
    gaussian_mean_class_zero_2d = get_gaussian_class_mean(train_class_zero_2d_data)
    gaussian_covariance_class_zero_2d = get_gaussian_covariance_matrix(
        train_class_zero_2d_data, gaussian_mean_class_zero_2d
    )
    print(f'Parameters of the Class Zero Gaussian Distribution: ')
    print(f'mean = {gaussian_mean_class_zero_2d}')
    print(f'covariance matrix = {gaussian_covariance_class_zero_2d}')
    print(f'covariance inverse = {np.linalg.inv(gaussian_covariance_class_zero_2d)}')
    print(f'covariance determinant = {np.linalg.det(gaussian_covariance_class_zero_2d)}')

    gaussian_mean_class_one_2d = get_gaussian_class_mean(train_class_one_2d_data)
    gaussian_covariance_class_one_2d = get_gaussian_covariance_matrix(
        train_class_one_2d_data, gaussian_mean_class_one_2d
    )
    print(f'Parameters of the Class One Gaussian Distribution: ')
    print(f'mean = {gaussian_mean_class_one_2d}')
    print(f'covariance matrix = {gaussian_covariance_class_one_2d}')
    print(f'covariance inverse = {np.linalg.inv(gaussian_covariance_class_one_2d)}')
    print(f'covariance determinant = {np.linalg.det(gaussian_covariance_class_one_2d)}')

    # Question 3: Use ML classifier on estimated class distributions to make predictions on test set. Find the
    # prediction error of the ML classifier.
    gaussian_ml_prediction_error = prediction_error(
        maximum_likelihood_classifier,
        gaussian_probability_distribution,
        test_data_2d,
        test_labels_2d,
        c_zero_mean=gaussian_mean_class_zero_2d,
        c_zero_covariance=gaussian_covariance_class_zero_2d,
        c_one_mean=gaussian_mean_class_one_2d,
        c_one_covariance=gaussian_covariance_class_one_2d
    )
    print(f'Gaussian Maximum Likelihood Prediction Error = {gaussian_ml_prediction_error}')

    # Question 4: Plot the decision boundary for the classifier.
    mathematical_decision_boundary(
        train_class_zero_2d_data,
        train_class_one_2d_data,
        gaussian_mean_class_zero_2d,
        gaussian_covariance_class_zero_2d,
        gaussian_mean_class_one_2d,
        gaussian_covariance_class_one_2d
    )
    decision_boundary(
        train_class_zero_2d_data,
        train_class_one_2d_data,
        gaussian_mean_class_zero_2d,
        gaussian_covariance_class_zero_2d,
        gaussian_mean_class_one_2d,
        gaussian_covariance_class_one_2d
    )

    # Question 5: Find prior probabilities for both classes.
    class_zero_prior = get_prior_probability(len(train_class_zero_2d_data), len(train_data_2d))
    print(f'p(C0) = {class_zero_prior}')
    class_one_prior = get_prior_probability(len(train_class_one_2d_data), len(train_data_2d))
    print(f'p(C1) = {class_one_prior}')

    # Question 6: Using the prior probabilities and estimated class distributions, make predictions on the test set
    # using the MAP classifier.
    gaussian_map_prediction_error = prediction_error(
        maximum_a_priori_classifier,
        gaussian_probability_distribution,
        test_data_2d,
        test_labels_2d,
        c_zero_mean=gaussian_mean_class_zero_2d,
        c_zero_covariance=gaussian_covariance_class_zero_2d,
        c_zero_prior=class_zero_prior,
        c_one_mean=gaussian_mean_class_one_2d,
        c_one_covariance=gaussian_covariance_class_one_2d,
        c_one_prior=class_one_prior
    )
    print(f'Gaussian Maximum A Priori Prediction Error = {gaussian_map_prediction_error}')


def exercise_2():
    print('Exercise 2')
    train_data_1d, train_labels_1d, test_data_1d, test_labels_1d = get_pca_transformed_dataset(1)

    train_class_zero_1d_data = get_class_zero(train_data_1d, train_labels_1d)
    train_class_one_1d_data = get_class_one(train_data_1d, train_labels_1d)

    # Question 3: Use the ML classifier on the estimated distributions. Find the prediction error for the three
    # different ML classifiers (three different probability distributions).

    # Exponential Distribution
    exponential_lambda_class_zero_1d = get_class_lambda(train_class_zero_1d_data)
    exponential_lambda_class_one_1d = get_class_lambda(train_class_one_1d_data)
    exponential_ml_prediction_error = prediction_error(
        maximum_likelihood_classifier,
        exponential_probability_distribution,
        test_data_1d,
        test_labels_1d,
        c_zero_lambda=exponential_lambda_class_zero_1d,
        c_one_lambda=exponential_lambda_class_one_1d
    )
    print(f'Exponential Prediction Error = {exponential_ml_prediction_error}')

    # Uniform Distribution
    uniform_a_class_zero_1d, uniform_b_class_zero_1d = get_class_parameters(train_class_zero_1d_data)
    uniform_a_class_one_1d, uniform_b_class_one_1d = get_class_parameters(train_class_one_1d_data)
    uniform_ml_prediction_error = prediction_error(
        maximum_likelihood_classifier,
        uniform_probability_distribution,
        test_data_1d,
        test_labels_1d,
        c_zero_a=uniform_a_class_zero_1d,
        c_zero_b=uniform_b_class_zero_1d,
        c_one_a=uniform_a_class_one_1d,
        c_one_b=uniform_b_class_one_1d
    )
    print(f'Uniform Prediction Error = {uniform_ml_prediction_error}')

    # Gaussian Distribution
    gaussian_mean_class_zero_1d = get_gaussian_class_mean(train_class_zero_1d_data)
    gaussian_covariance_class_zero_1d = get_gaussian_covariance_matrix(
        train_class_zero_1d_data, gaussian_mean_class_zero_1d
    )
    gaussian_mean_class_one_1d = get_gaussian_class_mean(train_class_one_1d_data)
    gaussian_covariance_class_one_1d = get_gaussian_covariance_matrix(
        train_class_one_1d_data, gaussian_mean_class_one_1d
    )
    gaussian_ml_prediction_error_1d = prediction_error(
        maximum_likelihood_classifier,
        gaussian_probability_distribution,
        test_data_1d,
        test_labels_1d,
        c_zero_mean=gaussian_mean_class_zero_1d,
        c_zero_covariance=gaussian_covariance_class_zero_1d,
        c_one_mean=gaussian_mean_class_one_1d,
        c_one_covariance=gaussian_covariance_class_one_1d
    )
    print(f'Gaussian Maximum Likelihood Prediction Error = {gaussian_ml_prediction_error_1d}')


if __name__ == '__main__':
    exercise_1()
    exercise_2()
