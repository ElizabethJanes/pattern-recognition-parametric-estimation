def prediction_error(
        classifier,
        probability_distribution,
        test_data,
        test_labels,
        c_zero_mean,
        c_zero_covariance,
        c_one_mean,
        c_one_covariance,
        c_zero_prior=None,
        c_one_prior=None
):
    incorrect_prediction_count = 0
    for test_vector, label in zip(test_data, test_labels):
        classification = classifier(
            test_vector,
            probability_distribution,
            c_zero_mean,
            c_zero_covariance,
            c_zero_prior,
            c_one_mean,
            c_one_covariance,
            c_one_prior
        )
        if classification != label:
            incorrect_prediction_count += 1
        # print(f'classification = {ml_classification}, label = {label}')

    return incorrect_prediction_count / len(test_data)
