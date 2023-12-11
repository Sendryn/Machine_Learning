# Author: Sandrine NEANG
# Date: 15/09/2023
# Project: 03 Sequential Estimation
# Acknowledgements: 
#


from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1234)

def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    return np.random.multivariate_normal(mean, np.power(var,2) * np.identity(k), n)


def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    return mu + (1/n) * (x - mu)
    

def _plot_sequence_estimate():
    data = gen_data(100, 3, np.array([0, 0, 0]), 4)
    estimates = [np.array([0, 0, 0])]
    
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i], data[i,:], i + 1))

    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    
    plt.legend(loc='upper center')
    plt.savefig('1_5_1.png')
    plt.show()


def _square_error(y, y_hat):
    return (y - y_hat)**2


def _plot_mean_square_error():
    #Ground truth y
    y = gen_data(100, 3, np.array([0,0,0]),4)
    
    #Estimate / Prediction y_hat (initilization)
    y_hat = [np.array([0,0,0])]
    
    #Mean Square Error calculation
    mean_square_error = []
    for i in range(y.shape[0]):
        y_hat.append(update_sequence_mean(y_hat[i], y[i,:], i + 1))
        #Let's take the mean of the y_hat three dimensions values to get the average error across all three dimensions
        mean_square_error.append(np.mean(_square_error(y_hat[i],np.array([0,0,0]))))
               
    plt.plot(mean_square_error)
    plt.savefig("1_6_1.png")
    plt.show()



# # Naive solution to the independent question.

# def gen_changing_data(
#     n: int,
#     k: int,
#     start_mean: np.ndarray,
#     end_mean: np.ndarray,
#     var: float
# ) -> np.ndarray:
#     # remove this if you don't go for the independent section
#     ...


# def _plot_changing_sequence_estimate():
#     # remove this if you don't go for the independent section
#     ...


if __name__ == '__main__':

    # np.random.seed(1234)
    # Section 1.1
    gen_data(2, 3, np.array([0, 1, -1]), 1.3)
    gen_data(5, 1, np.array([0.5]), 0.5)
    
    # Section 1.2
    X = gen_data(300, 3, np.array([0, 1, -1]), np.sqrt(3))
    scatter_3d_data(X)
    bar_per_axis(X)
    
    #Section 1.3 
    #Write in 1_2_1.txt
    #We can use a sequential estimate
    
    #Section 1.4
    mean = np.mean(X, 0)
    new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)
    mu = update_sequence_mean(mean, new_x, X.shape[0])
        
    #Section 1.5
    _plot_sequence_estimate()
    
    #Section 1.6
    _plot_mean_square_error()