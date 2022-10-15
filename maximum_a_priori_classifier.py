def get_prior_probability(class_training_count, total_training_count):
    prior = class_training_count/total_training_count
    return prior


def maximum_a_priori_classifier(
        test_vector,
        probability_distribution,
        c_zero_mean,
        c_zero_covariance,
        c_zero_prior,
        c_one_mean,
        c_one_covariance,
        c_one_prior):
    c_zero_probability_distribution = probability_distribution(c_zero_mean, c_zero_covariance, test_vector)
    c_one_probability_distribution = probability_distribution(c_one_mean, c_one_covariance, test_vector)

    if c_zero_probability_distribution*c_zero_prior > c_one_probability_distribution*c_one_prior:
        classification = 0
    else:
        classification = 1

    return classification
