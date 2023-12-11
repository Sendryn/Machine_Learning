# Author: Sandrine NEANG
# Date: 29/09/2023
# Project: 05 Classfication
# Acknowledgements: 
#


from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from sklearn.metrics import accuracy_score


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    # We filter the features which belong to selected_class
    class_features = []
    for i in range(len(features)):
        if (targets[i]==selected_class):
            class_features.append(features[i])
            
    means = []
    class_features = np.array(class_features)
    for j in range(features.shape[1]):
        means.append(np.mean(class_features[:,j]))
    
    return np.array(means)


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    # We filter the features which belong to selected_class
    class_features = []
    for i in range(len(features)):
        if (targets[i]==selected_class):
            class_features.append(features[i])

    # Calculate the covariance of the selected_class features
    class_cov = np.cov(class_features, rowvar=False)

    return class_cov


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    return multivariate_normal(class_mean, class_covar).pdf(feature)


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))
    
    likelihoods = []
    for i in range(test_features.shape[0]):
        likelihoods.append([])
        for cl in classes:
            likelihoods[i].append(likelihood_of_class(test_features[i, :], means[classes.index(cl)], covs[classes.index(cl)]))
    return np.array(likelihoods)
    

def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    
    return np.argmax(likelihoods, axis=1)


def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs, prior = [], [], []
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))
        prior.append(np.sum(train_targets == class_label) / len(train_targets))
    
    # Calculating the likelihoods
    likelihoods = []
    for i in range(test_features.shape[0]):
        likelihoods.append([])
        for c in classes:
            likelihoods[i].append(likelihood_of_class(test_features[i, :], means[classes.index(c)], covs[classes.index(c)]))

    for i in range(len(likelihoods)):
        for p in range(len(likelihoods[i])):
            likelihoods[i][p] = likelihoods[i][p] * prior[p]

    return np.array(likelihoods)


def confusion_matrix(prediction, targets):
    confusion_matrix = np.zeros((len(prediction), len(prediction)), dtype=int)
    
    for actual, predicted in zip(targets, prediction):
       confusion_matrix[actual, predicted] += 1
        
    return confusion_matrix


if __name__ == '__main__':
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets)\
    = split_train_test(features, targets, train_ratio=0.6)

    #Section 1.1
    mean_of_class(train_features, train_targets, 0)
    
    #Section 1.2
    covar_of_class(train_features, train_targets, 0)
    
    #Section 1.3
    class_mean = mean_of_class(train_features, train_targets, 0)
    class_cov = covar_of_class(train_features, train_targets, 0)
    likelihood_of_class(test_features[0, :], class_mean, class_cov)
    
    #Section 1.4
    maximum_likelihood(train_features, train_targets, test_features, classes)

    #Section 1.5
    likelihoods1 = maximum_likelihood(train_features, train_targets, test_features, classes)
    predictions1 = predict(likelihoods1)
    
    #Section 2.1
    likelihoods2 = maximum_aposteriori(train_features, train_targets, test_features, classes)
    predictions2 = predict(likelihoods2)
    
    #Section 2.2
    accuracyscore1 = accuracy_score(test_targets, predictions1)
    accuracyscore2 = accuracy_score(test_targets, predictions2)
    
    print(confusion_matrix(predictions1, test_targets))
    print(confusion_matrix(predictions2, test_targets))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    