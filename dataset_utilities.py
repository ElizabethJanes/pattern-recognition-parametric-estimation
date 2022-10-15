import os
import numpy as np
from torchvision import datasets
from sklearn.decomposition import PCA


def get_pca_transformed_dataset(pca_components):
    project_dir = os.getcwd()
    train_data_dir = os.path.join(project_dir, 'train')
    train_dataset = datasets.MNIST(root=train_data_dir, train=True, download=True)
    train_data = train_dataset.data.numpy()
    train_labels = train_dataset.targets.numpy()

    test_data_dir = os.path.join(project_dir, 'test')
    test_dataset = datasets.MNIST(root=test_data_dir, train=False, download=True)
    test_data = test_dataset.data.numpy()
    test_labels = test_dataset.targets.numpy()

    target_train_indices = np.where((train_labels == 0) | (train_labels == 1))
    target_train_data = train_data[target_train_indices]
    target_train_labels = train_labels[target_train_indices]

    target_test_indices = np.where((test_labels == 0) | (test_labels == 1))
    target_test_data = test_data[target_test_indices]
    target_test_labels = test_labels[target_test_indices]

    reshaped_train_data = np.reshape(target_train_data, (len(target_train_data), 784))
    reshaped_test_data = np.reshape(target_test_data, (len(target_test_data), 784))

    pca = PCA(n_components=pca_components)
    pca_reshaped_train_data = pca.fit_transform(reshaped_train_data)
    pca_reshaped_test_data = pca.transform(reshaped_test_data)

    return pca_reshaped_train_data, target_train_labels, pca_reshaped_test_data, target_test_labels


def get_class_zero(data, labels):
    class_zero_indices = np.where(labels == 0)
    class_zero_data = data[class_zero_indices]
    return class_zero_data


def get_class_one(data, labels):
    class_one_indices = np.where(labels == 1)
    class_one_data = data[class_one_indices]
    return class_one_data
