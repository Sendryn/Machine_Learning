# Author: Sandrine NEANG
# Date: 29/09/2023
# Project: 06 PCA
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tools import load_cancer


def standardize(X: np.ndarray) -> np.ndarray:
    '''
    Standardize an array of shape [N x 1]

    Input arguments:
    * X (np.ndarray): An array of shape [N x 1]

    Returns:
    (np.ndarray): A standardized version of X, also
    of shape [N x 1]
    '''
    mean = np.mean(X)
    std = np.std(X)

    return (X - mean) / std


def scatter_standardized_dims(
    X: np.ndarray,
    i: int,
    j: int,
):
    '''
    Plots a scatter plot of N points where the n-th point
    has the coordinate (X_ni, X_nj)

    Input arguments:
    * X (np.ndarray): A [N x f] array
    * i (int): The first index
    * j (int): The second index
    '''
    X_standardized = standardize(X)
    X_ni = X_standardized[:,i]
    X_nj = X_standardized[:,j]
    
    plt.scatter(X_ni,X_nj)


def _scatter_cancer():
    X, y = load_cancer()
    
    plt.figure(figsize=(30, 20))
    for i in range(30):
        plt.subplot(5, 6, i+1)
        scatter_standardized_dims(X, 0, i)
    

def _plot_pca_components():
    X, y = load_cancer()
    X = standardize(X)
    
    pca = PCA()
    pca.fit_transform(X)
    
    plt.figure(figsize=(30, 20))
    for i in range(30):
        plt.subplot(5, 6, i+1)
        plt.plot(pca.components_[i])
        plt.title("PCA " + str(i+1))
    plt.show()
    

def _plot_eigen_values():
    X, y = load_cancer()
    X = standardize(X)
    
    pca = PCA()
    pca.fit_transform(X)
    
    plt.plot(pca.explained_variance_)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()


def _plot_log_eigen_values():
    X, y = load_cancer()
    X = standardize(X)
    
    pca = PCA()
    pca.fit_transform(X)
    
    plt.plot(np.log(pca.explained_variance_))
    plt.xlabel('Eigenvalue index')
    plt.ylabel('$\log_{10}$ Eigenvalue')
    plt.grid()
    plt.show()


def _plot_cum_variance():
    X, y = load_cancer()
    X = standardize(X)
    
    pca = PCA()
    pca.fit_transform(X)
    
    plt.plot(np.cumsum(pca.explained_variance_))
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Percentage variance')
    plt.grid()
    plt.show()


# if __name__ == '__main__':
#     #Section 1.1
#     standardize(np.array([[0, 0], [0, 0], [1, 1], [1, 1]]))
    
#     #Section 1.2
#     X = np.array([
#     [1, 2, 3, 4],
#     [0, 0, 0, 0],
#     [4, 5, 5, 4],
#     [2, 2, 2, 2],
#     [8, 6, 4, 2]])
#     #scatter_standardized_dims(X, 0, 2)
#     #plt.show()
    
#     #Section 1.3
#     #_scatter_cancer()
    
#     #Section 2.1
#     #_plot_pca_components()
    
#     #Section 3.1
#     #_plot_eigen_values()
    
#     #Section 3.2
#     #_plot_log_eigen_values()
    
#     #Section 3.3
#     #_plot_cum_variance()