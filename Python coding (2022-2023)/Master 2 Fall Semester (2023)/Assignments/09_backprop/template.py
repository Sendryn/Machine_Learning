from typing import Union
import numpy as np

from tools import load_iris, split_train_test
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    x = np.clip(x, -100, None)
    return (1 / (1 + np.exp(-x)))


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    return sigmoid(x) * (1 - sigmoid(x))


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    weighted_sum = np.dot(x, w)
    result = sigmoid(weighted_sum)
    return weighted_sum, result


def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    z0 = np.insert(x, 0, 1.0)
    
    a1, z1 = [], [1.0]
    for i in range(np.shape(W1)[1]):
        a_i, z_i = perceptron(z0, W1[:,i])
        a1.append(a_i)
        z1.append(z_i)

    a2, y = [], []
    for i in range(np.shape(W2)[1]):
        a_i, y_i = perceptron(z1, W2[:,i])
        a2.append(a_i)
        y.append(y_i)
    
    return np.array(y), np.array(z0), np.array(z1), np.array(a1), np.array(a2) 


def backprop(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    #1
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    
    #2
    delta_k = y - target_y
    
    #3
    delta_j= np.dot(W2[1:, :], delta_k) * d_sigmoid(a1)
    
    #4
    dE1 = np.zeros_like(W1)
    dE2 = np.zeros_like(W2)
    
    #5
    for i in range(np.shape(dE1)[0]):
        for j in range(np.shape(dE1)[1]):
            dE1[i,j] = delta_j[j] * z0[i]
    
    for j in range(np.shape(dE2)[0]):
        for k in range(np.shape(dE2)[1]):
            dE2[j,k] = delta_k[k] * z1[j] 
            
    return y, dE1, dE2


def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    #1
    E_total = []
    misclassification_rate = []
    N = X_train.shape[0]
    guesses = []
    
    #2
    for _ in range(iterations):
        
        #3
        dE1_total = np.zeros_like(W1)
        dE2_total = np.zeros_like(W2)
        error_total = 0
        misclassifications = 0
        
        #4
        for i in range(N):
            x = X_train[i, :]
            target_y = np.zeros(K)
            target_y[t_train[i]] = 1.0
            
            y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
            dE1_total += dE1
            dE2_total += dE2
            #6
            error_total -= np.sum(target_y * np.log(y) + (1 - target_y) * np.log(1 - y))

            if np.argmax(y) != t_train[i]: misclassifications += 1
        
        #5
        W1 = W1 - eta * dE1_total / N
        W2 = W2 - eta * dE2_total / N

        misclassification_rate.append(misclassifications/N)
        E_total.append(error_total/N)
    
    for j in range(N):
        y, z0, z1, a1, a2 = ffnn(X_train[j, :], M, K, W1, W2)
        guesses.append(np.argmax(y))
    #7
    return W1, W2, E_total, misclassification_rate, np.array(guesses, dtype=float)


def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    N = X.shape[0]
    guesses = np.zeros(N, dtype=int)
    
    for i in range(N):
      y, z0, z1, a1, a2 = ffnn(X[i, :], M, K, W1, W2)
      guesses[i] = np.argmax(y)
   
    return guesses



if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
    
    #Section 1.1
    print(sigmoid(0.5))
    print(d_sigmoid(0.2))
    
    
    #Section 1.2
    print(perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1])))
    print(perceptron(np.array([0.2,0.4]),np.array([0.1,0.4])))
    
    
    #Section 1.3
    # initialize the random generator to get repeatable results
    np.random.seed(1234)
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets)
    # initialize the random generator to get repeatable results
    np.random.seed(1234)
    # Take one point:
    x = train_features[0, :]
    K = 3 # number of classes
    M = 10
    D = 4
    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    print("y: ", y)
    print("z0: ", z0)
    print("z1: ",z1)
    print("a1: ",a1)
    print("a2: ",a2)
    
    
    #Section 1.4
    # initialize random generator to get predictable results
    np.random.seed(42)
    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]
    x = features[0, :]
    # create one-hot target for the feature
    target_y = np.zeros(K)
    target_y[targets[0]] = 1.0
    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
    print("y: ", y)
    print("dE1: ", dE1)
    print("dE2: ", dE2)
    
    
    #Section 2.1
    # initialize the random seed to get predictable results
    np.random.seed(1234)
    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]
    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)
    print("W1tr: ", W1tr)
    print("W2tr: ", W2tr)
    print("Etotal: ", Etotal)
    print("misclassification_rate: ", misclassification_rate)
    print("last_guesses: ", last_guesses)
    
    
    #Section 2.3
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(train_features[:, :], train_targets[:], M, K, W1, W2, 500, 0.1)  #training on 80% of the dataset
    
    predictions = test_nn(test_features, M, K, W1tr, W2tr)
    
    print("predictions: ", predictions)
    print("test_targets: ",test_targets)
    
    accuracy = np.count_nonzero(last_guesses == train_targets)/len(train_targets)*100
    print("\naccuracy: ", accuracy)
    confusion = confusion_matrix(test_targets, np.array(predictions))
    print("\nConfusion Matrix:\n", confusion)
    
    plt.plot(Etotal)
    plt.xlabel("Iterations")
    plt.ylabel("E_total")
    plt.show()
    
    plt.plot(misclassification_rate)
    plt.xlabel("Iterations")
    plt.ylabel("Misclassification rate")
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    