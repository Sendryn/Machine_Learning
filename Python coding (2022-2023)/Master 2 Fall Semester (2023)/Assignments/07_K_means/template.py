# Author: Sandrine Neang
# Date: 13/10/2023
# Project: 07 K_means
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union

from tools import load_iris, image_to_numpy, plot_gmm_results


def distance_matrix(
    X: np.ndarray,
    Mu: np.ndarray
) -> np.ndarray:
    '''
    Returns a matrix of euclidian distances between points in
    X and Mu.

    Input arguments:
    * X (np.ndarray): A [n x f] array of samples
    * Mu (np.ndarray): A [k x f] array of prototypes

    Returns:
    out (np.ndarray): A [n x k] array of euclidian distances
    where out[i, j] is the euclidian distance between X[i, :]
    and Mu[j, :]
    '''    
    
    X1,X2 = X.shape
    Mu1,Mu2 = Mu.shape

    # Initialize the distances matrix
    out = np.zeros((X1, Mu1))

    # Calculate Euclidean distances using a for loop
    for i in range(X1):
        for j in range(Mu1):
            out[i, j] = np.sqrt(np.sum((X[i, :] - Mu[j, :]) ** 2))

    return out


def determine_r(dist: np.ndarray) -> np.ndarray:
    '''
    Returns a matrix of binary indicators, determining
    assignment of samples to prototypes.

    Input arguments:
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    out (np.ndarray): A [n x k] array where out[i, j] is
    1 if sample i is closest to prototype j and 0 otherwise.
    '''
    
    n, k = dist.shape

    # Initialize the output matrix
    out = np.zeros((n, k), dtype=int)

    # Iterate through each sample
    for i in range(n):
        # Find the index of the minimum distance
        min_index = 0
        min_distance = dist[i, 0]
        for j in range(1, k):
            if dist[i, j] < min_distance:
                min_index = j
                min_distance = dist[i, j]

        # Set the corresponding entry to 1
        out[i, min_index] = 1

    return out


def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    '''
    Calculates the value of the objective function given
    arrays of indicators and distances.

    Input arguments:
    * R (np.ndarray): A [n x k] array where out[i, j] is
        1 if sample i is closest to prototype j and 0
        otherwise.
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    * out (float): The value of the objective function
    '''
    n,k = R.shape
    out = np.zeros(k)
    
    for i in range(n):
        for j in range(k):
            out[j] += R[i,j] * dist[i,j]
            
    return (1/n)*sum(out)


def update_Mu(
    Mu: np.ndarray,
    X: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    '''
    Updates the prototypes, given arrays of current
    prototypes, samples and indicators.

    Input arguments:
    Mu (np.ndarray): A [k x f] array of current prototypes.
    X (np.ndarray): A [n x f] array of samples.
    R (np.ndarray): A [n x k] array of indicators.

    Returns:
    out (np.ndarray): A [k x f] array of updated prototypes.
    '''
    
    k, f = Mu.shape
    n = X.shape[0]

    # Initialize the updated prototypes
    out = np.zeros((k, f))

    # Update each prototype
    for j in range(k):
        numerator = np.zeros(f)
        denominator = 0.0

        # Calculate the numerator and denominator
        for i in range(n):
            numerator += R[i, j] * X[i, :]
            denominator += R[i, j]

        # Avoid division by zero by checking if denominator is zero
        if denominator != 0:
            out[j, :] = numerator / denominator
        else:
            # If no samples are assigned to the prototype, keep the current value
            out[j, :] = Mu[j, :]

    return out


def k_means(
    X: np.ndarray,
    k: int,
    num_its: int
) -> Union[list, np.ndarray, np.ndarray]:
    # We first have to standardize the samples
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_st, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]

    # !!! Your code here !!!
    objective_values = []
    for j in range(num_its):
        dist = distance_matrix(X_standard, Mu)
        R = determine_r(dist)
        Mu = update_Mu(Mu, X_standard, R)
        
        objective_value = determine_j(R, dist)
        objective_values.append(objective_value)
        
    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean

    # !!! Your code here !!!
    
    return Mu, R, objective_values


def _plot_j():
    X, y, c = load_iris()
    k, num_its = 4, 10
    # We first have to standardize the samples
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_st, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]

    objective_values = []
    for j in range(num_its):
        dist = distance_matrix(X_standard, Mu)
        R = determine_r(dist)
        Mu = update_Mu(Mu, X_standard, R)
        
        objective_value = determine_j(R, dist)
        objective_values.append(objective_value)
        
    plt.plot(objective_values, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('Objective Function Progression')
    plt.show()


def _plot_multi_j():
    X, y, c = load_iris()
    k2, k3, k5, k10, num_its = 2, 3, 5, 10, 10
    # We first have to standardize the samples
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_st, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu2 = X_standard[nn[0: k2], :]
    Mu3 = X_standard[nn[0: k3], :]
    Mu5 = X_standard[nn[0: k5], :]
    Mu10 = X_standard[nn[0: k10], :]

    objective_values2 = []
    objective_values3 = []
    objective_values5 = []
    objective_values10 = []
    
    for j in range(num_its):
        dist2 = distance_matrix(X_standard, Mu2)
        R2 = determine_r(dist2)
        Mu2 = update_Mu(Mu2, X_standard, R2)
        objective_value2 = determine_j(R2, dist2)
        objective_values2.append(objective_value2)
        
        dist3 = distance_matrix(X_standard, Mu3)
        R3 = determine_r(dist3)
        Mu3 = update_Mu(Mu3, X_standard, R3)
        objective_value3 = determine_j(R3, dist3)
        objective_values3.append(objective_value3)
        
        dist5 = distance_matrix(X_standard, Mu5)
        R5 = determine_r(dist5)
        Mu5 = update_Mu(Mu5, X_standard, R5)
        objective_value5 = determine_j(R5, dist5)
        objective_values5.append(objective_value5)
        
        dist10 = distance_matrix(X_standard, Mu10)
        R10 = determine_r(dist10)
        Mu10 = update_Mu(Mu10, X_standard, R10)
        objective_value10 = determine_j(R10, dist10)
        objective_values10.append(objective_value10)
        
          
    plt.plot(objective_values2, label='k2')
    plt.plot(objective_values3, label='k3')
    plt.plot(objective_values5, label='k5')
    plt.plot(objective_values10, label='k10')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('Objective Function Progression')
    plt.legend()
    plt.show()


def k_means_predict(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int
) -> np.ndarray:
    '''
    Determine the accuracy and confusion matrix
    of predictions made by k_means on a dataset
    [X, t] where we assume the most common cluster
    for a class label corresponds to that label.

    Input arguments:
    * X (np.ndarray): A [n x f] array of input features
    * t (np.ndarray): A [n] array of target labels
    * classes (list): A list of possible target labels
    * num_its (int): Number of k_means iterations

    Returns:
    * the predictions (list)
    '''
    # Run k-means clustering
    Mu, final_R = k_means(X, len(classes), num_its)

    # Map cluster indices to class labels
    cluster_to_class = {}
    for i, class_label in enumerate(classes):
        # Find the most common cluster for each class label
        most_common_cluster = np.argmax(np.sum(final_R[t == class_label], axis=0))
        cluster_to_class[most_common_cluster] = class_label

    # Assign class labels to clusters
    predictions = [cluster_to_class[cluster] for cluster in np.argmax(final_R, axis=1)]

    return predictions


def _iris_kmeans_accuracy():
    # Load the Iris dataset
    X, y, c = load_iris()

    # Run k-means clustering
    num_iterations = 10  # Adjust the number of iterations as needed
    predictions = k_means_predict(X, y, c, num_iterations)

    # Calculate accuracy
    accuracy = accuracy_score(y, predictions)

    # Calculate confusion matrix
    cm = confusion_matrix(y, predictions)

    return accuracy, cm


def _my_kmeans_on_image():
    pass


def plot_image_clusters(n_clusters: int):
    '''
    Plot the clusters found using sklearn k-means.
    '''
    image, (w, h) = image_to_numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(image.reshape(-1, 3))  
    plt.subplot(121)
    plt.imshow(image.reshape(w, h, 3))
    plt.subplot(122)
    # uncomment the following line to run
    plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")
    plt.show()
    
# plot_image_clusters(2)
# plot_image_clusters(5)
# plot_image_clusters(10)
# plot_image_clusters(20)