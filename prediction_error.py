def prediction_error(
        classifier,
        probability_distribution,
        test_data,
        test_labels,
        c_zero_mean=None,
        c_zero_covariance=None,
        c_zero_prior=None,
        c_zero_lambda=None,
        c_zero_a=None,
        c_zero_b=None,
        c_one_mean=None,
        c_one_covariance=None,
        c_one_prior=None,
        c_one_lambda=None,
        c_one_a=None,
        c_one_b=None
):
    """
    :param classifier: The function for the classifier to be used for making predictions on the test dataset (ML or MAP)
    :param probability_distribution: The probability distribution to be used to evaluate the likelihood of each test point's classification
    :param test_data: The full MNIST test dataset for the classes of interest (class 0 and 1)
    :param test_labels: The MNIST labels corresponding to the test dataset
    :param c_zero_mean: Gaussian mean for class zero; must be included if the Gaussian probability distribution is used
    :param c_zero_covariance: Gaussian covariance for class zero; must be included if the Gaussian probability distribution is used
    :param c_zero_prior: Prior probability of class zero; must be included if the MAP classifier is used
    :param c_zero_lambda: Lambda parameter for class zero Exponential distribution; must be included if the Exponential probability distribution is used
    :param c_zero_a: Parameter a for class zero, to be included if the Uniform distribution is used
    :param c_zero_b: Parameter b for class zero, to be included if the Uniform distribution is used
    :param c_one_mean: Gaussian mean for class one; must be included if the Gaussian probability distribution is used
    :param c_one_covariance: Gaussian covariance for class one; must be included if the Gaussian probability distribution is used
    :param c_one_prior: Prior probability of class one; must be included if the MAP classifier is used
    :param c_one_lambda: Lambda parameter for class one Exponential distribution; must be included if the Exponential probability distribution is used
    :param c_one_a: Parameter a for class one, to be included if the Uniform distribution is used
    :param c_one_b: Parameter b for class one, to be included if the Uniform distribution is used
    :return: Prediction error for the given classifier using the provided probability distribution with the test dataset
    """
    incorrect_prediction_count = 0
    for test_vector, label in zip(test_data, test_labels):
        classification = classifier(
            test_vector,
            probability_distribution,
            c_zero_mean,
            c_zero_covariance,
            c_zero_prior,
            c_zero_lambda,
            c_zero_a,
            c_zero_b,
            c_one_mean,
            c_one_covariance,
            c_one_prior,
            c_one_lambda,
            c_one_a,
            c_one_b
        )
        if classification != label:
            incorrect_prediction_count += 1

    return incorrect_prediction_count / len(test_data)
