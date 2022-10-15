def maximum_likelihood_classifier(
        test_vector,
        probability_distribution,
        c_zero_mean,
        c_zero_covariance,
        c_zero_prior,
        c_one_mean,
        c_one_covariance,
        c_one_prior
):
    c_zero_probability_distribution = probability_distribution(c_zero_mean, c_zero_covariance, test_vector)
    c_one_probability_distribution = probability_distribution(c_one_mean, c_one_covariance, test_vector)

    if c_zero_probability_distribution > c_one_probability_distribution:
        classification = 0
    else:
        classification = 1

    return classification
