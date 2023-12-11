# Author: Sandrine NEANG
# Date: 31/08/2023
# Project: 02_nearest_neighbours
# Acknowledgements: 
#

import numpy as np
import matplotlib.pyplot as plt
from help import remove_one
from tools import load_iris, split_train_test, plot_points


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    euclid_sum = 0
    for i in range(len(x)): euclid_sum += (x[i] - y[i])**2    
    return np.sqrt(euclid_sum)


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x,points[i])
    return distances


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    # Get the indices that would sort the distances in ascending order
    tab = np.argsort(euclidian_distances(x, points))
    # Return the first k indices (corresponding to the k-nearest neighbors)
    return tab[0:k]


def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    #this function return the most common class label among the k nearest points
    
    #count will store the count of occurrences for each class in the 'classes' list
    count = []
    for i in range(len(classes)):
        #n will count the occurrence of each class
        n = 0
        for j in range(len(targets)):
            if classes[i] == targets[j]: n = n + 1
        count.append(n)
    #Return the class which has the maximum of occurrence
    return classes[count.index(max(count))]    


def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    # Get the indices of the k nearest neighbors of the test point x
    k_nearest_indices = k_nearest(x, points, k)
    
    # Create a list 'k_nearest_classes' to store the classes of the k nearest neighbors
    k_nearest_classes = []
    
    for i in k_nearest_indices:
        k_nearest_classes.append(point_targets[i])
        
    # Predict the class of the test point based on the classes of its k nearest neighbors
    predicted_class = vote(k_nearest_classes, classes)
    return predicted_class
    

def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    
    # List to store predicted class labels for each data point
    predicted_classes = []
    
    for i in range(len(points)):
        # The current data point is excluded from the training set and its corresponding target
        modified_points = remove_one(points, i)
        modified_points_targets = remove_one(point_targets, i)
        
        # Prediction of the class label of the current data point by using k-nearest neighbors function
        prediction = knn(points[i], modified_points, modified_points_targets, classes, k)
        predicted_classes.append(prediction)
        
    return predicted_classes
    

def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    
    predicted_classes = knn_predict(points, point_targets, classes, k)
    
    correct_predictions = 0    
    total_data_points = len(point_targets)
    
    for i in range(total_data_points):
        if predicted_classes[i] == point_targets[i]:
            correct_predictions += 1

    return correct_predictions/total_data_points
    
        
def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:

    predicted_classes = knn_predict(points, point_targets, classes, k)
    num_classes = len(classes)
    #Create a matrix filled with zeros where its dimension is according of the number of classes
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for true_label, pred_label in zip(point_targets, predicted_classes):
        if point_targets[true_label] == predicted_classes[pred_label]:
            # Rows represent predicted labels and columns represent true labels
            cm[pred_label][true_label] += 1

    return cm


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    
    klist = []
    accuracies = []
    for k in range(1, len(points)-1):
        #test values of k
        klist.append(k)
        #evaluation of each k value
        accuracies.append(knn_accuracy(points, point_targets, classes, k))
    
    return klist[accuracies.index(max(accuracies))]


def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    knn_predictions = knn_predict(points, point_targets, classes, k)
    
    colors = ['yellow', 'purple', 'blue']
            
    conditions = ["green" if knn_predictions[i] == point_targets[i] else "red" for i in range(len(knn_predictions))]
    
    for i in range(points.shape[0]):
        [x, y] = points[i,:2]
        plt.scatter(x, y, c=colors[point_targets[i]], edgecolors=conditions[i], linewidths=2)
    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.savefig('2_5_1.png')
    plt.show()



# if __name__ == '__main__':
#     d, t, classes = load_iris()
    #plot_points(d, t)
    
#     #Part 1.1
#     x, points = d[0,:], d[1:, :]
#     x_target, point_targets = t[0], t[1:] 
    
#     euclidian_distance(x, points[0])
#     euclidian_distance(x, points[50])
    
#     #Part 1.2
#     euclidian_distances(x, points)
    
#     #Part 1.3
#     k_nearest(x, points, 1)
#     k_nearest(x, points, 3)
    
#     #Part 1.4
#     vote(np.array([0,0,1,2]), np.array([0,1,2]))
#     vote(np.array([1,1,1,1]), np.array([0,1]))
    
#     #Part 1.5
#     knn(x, points, point_targets, classes, 1)
#     knn(x, points, point_targets, classes, 5)
#     knn(x, points, point_targets, classes, 150)
    
    # #Part 2.1
    # d, t, classes = load_iris()
    # (d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)
#     predictions = knn_predict(d_test, t_test, classes, 10)
#     predictions = knn_predict(d_test, t_test, classes, 5)
    
#     #Part 2.2
#     knn_accuracy(d_test, t_test, classes, 10)
#     knn_accuracy(d_test, t_test, classes, 5)
    
    # #Part 2.3
    # knn_confusion_matrix(d_test, t_test, classes, 10)
    # knn_confusion_matrix(d_test, t_test, classes, 20)
    
#     #Part 2.4
#     best_k(d_train, t_train, classes)
    
    # #Part 2.5
    # knn_plot_points(d, t, classes, 3)
    
    
    