# Author: Sandrine NEANG
# Date: 15/09/2023
# Project: 04 Linear Regression
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

import numpy as np
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal


def mvn_basis(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float
) -> np.ndarray:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * sigma: All normal distributions are isotropic with sigma*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    # Get the number of data vectors (N) and multivariate normal distributions (M)
    N, _ = features.shape
    M, D = mu.shape

    # Initialize the basis function matrix
    fi = np.zeros((N, M))

    # Calculate the covariance matrix (sigma*I)
    covariance_matrix = sigma * np.identity(D)

    # Calculate the basis functions :
    for i in range(N):
        for j in range(M):
            mvn = multivariate_normal.pdf(features[i], mu[j], covariance_matrix)
            fi[i, j] = mvn

    return fi


def _plot_mvn():
    X, t = load_regression_iris()
    N, D = X.shape
    M, sigma = 10, 10
    mu = np.zeros((M, D))
    for i in range(D):
        mmin = np.min(X[i, :])
        mmax = np.max(X[i, :])
        mu[:, i] = np.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, sigma)
    plt.plot(fi)
    plt.savefig("plot_1_2_1.png")
    plt.show()
    

def max_likelihood_linreg(
    fi: np.ndarray,
    targets: np.ndarray,
    lamda: float
) -> np.ndarray:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    
    Phi = np.dot(fi.T, fi)
    w = np.dot(np.linalg.inv(np.add(Phi, lamda * np.identity(len(Phi)))), np.transpose(fi))
    w = np.dot(w, targets)

    return w
    
    
def linear_model(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    w: np.ndarray
) -> np.ndarray:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    
    fi = mvn_basis(features, mu, sigma)
    output = np.dot(w, fi.T)

    return output
    

# plt.plot(prediction)
# plt.plot(t) #actual values
# plt.legend(['Prediction','Actual values'])
# plt.savefig('1_5_a.png')

# mean = []
# for i in range(len(prediction)):
#     mean.append((t[i] - prediction[i]) ** 2)

# plt.plot(mean)
# MSE = sum(mean)/len(prediction)
# print(MSE)
# plt.savefig('1_5_b')
# plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
