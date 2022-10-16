def maximum_likelihood_classifier(
        test_vector,
        probability_distribution,
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
    c_zero_probability_distribution = probability_distribution(
        test_vector, c_zero_mean, c_zero_covariance, c_zero_lambda, c_zero_a, c_zero_b
    )
    c_one_probability_distribution = probability_distribution(
        test_vector, c_one_mean, c_one_covariance, c_one_lambda, c_one_a, c_one_b
    )

    if c_zero_probability_distribution > c_one_probability_distribution:
        classification = 0
    else:
        classification = 1

    return classification
