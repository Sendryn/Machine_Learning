"""
@author: Sandrine Neang
@date: 07/10/2023
"""

import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

def split_train_test(features: np.ndarray, targets: np.ndarray,
    train_ratio:float=0.8) -> Union[tuple, tuple]:
    '''
    Shuffle the features and targets in unison and return
    two tuples of datasets, first being the training set,
    where the number of items in the training set is according
    to the given train_ratio
    '''
    np.random.seed(888)
    p = np.random.permutation(features.shape[0])
    features = features[p]
    targets = targets[p]

    split_index = int(features.shape[0] * train_ratio)

    train_features, train_targets = features[0:split_index, :],\
    targets[0:split_index]
    test_features, test_targets = features[split_index:-1, :],\
        targets[split_index: -1]

    return (train_features, train_targets), (test_features, test_targets)

def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    class_counts = np.zeros(len(classes))
    
    for target in targets:
        class_counts[target] += 1
    
    total_samples = len(targets)
    class_probabilities = class_counts / total_samples
    
    return class_probabilities

def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    split1 = features[:, split_feature_index] < theta
    split2 = features[:, split_feature_index] >= theta
    
    features_1 = features[split1]
    targets_1 = targets[split1]

    features_2 = features[split2]
    targets_2 = targets[split2]

    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    
    square = prior(targets, classes)
    for n in range(len(square)): square[n] = square[n]**2

    return 1/2 * (1 - sum(square))
        

def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1,classes)
    g2 = gini_impurity(t2,classes)
    n = t1.shape[0] + t2.shape[0]
    
    i = (t1.shape[0] * g1 + t2.shape[0] * g2) / n
    return i


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    
    (features_1, targets_1),(features_2, targets_2) = split_data(features, targets, split_feature_index, theta)
    weight = weighted_impurity(targets_1, targets_2, classes)
    
    return weight


def brute_best_split( 
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    
    best_gini, best_dim, best_theta = float("inf"), None, None
    
    # iterate feature dimensions
    for i in range(features.shape[1]):
        features_i = features[:,i]
        
        # create the thresholds
        thetas = np.linspace(features_i.min(), features_i.max(), num_tries+2)[1:-1]
        
        # iterate thresholds
        for theta in thetas:
            gini = total_gini_impurity(features, targets, classes, i, theta)
            if best_gini > gini: 
                best_gini = gini
                best_dim = i
                best_theta = theta
                
    return best_gini, best_dim, best_theta


class BreastCancerTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = ['M','B'],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Breast Cancer dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        return self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        return accuracy_score(self.test_targets, self.tree.predict(self.test_features))

    def plot(self):
        plt.figure(figsize=(10,10), dpi=300)        
        plot_tree(self.tree, filled=True, class_names=self.classes, rounded=True)
        plt.show()
        

    def guess(self):
        return self.tree.predict(self.test_features)
    
    def confusion_matrix(self):
        predictions = self.guess()
        num_classes = len(self.classes)
        cm = np.zeros((num_classes, num_classes), dtype=int)

        for true_label, pred_label in zip(self.test_targets, predictions):
            cm[true_label][pred_label] += 1

        return cm
        
        
import seaborn as sns

def feature_correlation_map(dataframe):
    # Calculate the correlation matrix
    corr_matrix = dataframe.corr()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))

    # Draw the heatmap with the mask
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": 0.5})

    plt.show()


def drop_highly_correlated_features(dataframe, threshold=0.5):
    # Calculate the correlation matrix
    corr_matrix = dataframe.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop the highly correlated features
    dataframe_dropped = dataframe.drop(to_drop, axis=1)

    return dataframe_dropped

        
def load_data():
    BreastCancerDataset = pd.read_csv('BreastCancerDataset.csv').drop(['Unnamed: 32'], axis=1)
    BreastCancerFeatures = BreastCancerDataset.drop(['id', 'diagnosis'], axis=1).values
    BreastCancerTargets = BreastCancerDataset['diagnosis'].values
    BreastCancerClasses = BreastCancerDataset['diagnosis'].unique()
    return BreastCancerFeatures, BreastCancerTargets, BreastCancerClasses


if __name__ == '__main__':
    features, targets, classes = load_data()
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, 15, 4.65)
    len(f_1)
    len(f_2)
    #total_gini_impurity(features, targets, classes, 2, 4.65)
    dt = BreastCancerTreeTrainer(features, targets, classes=classes)
    
    dt.train()
    dt.plot()
    print(f'The accuracy is: {dt.accuracy()}')
    print(f'I guessed: {dt.guess()}')
    print(f'The true targets are: {dt.test_targets}')
    #print(dt.confusion_matrix())
    
    BreastCancerDataset = pd.read_csv('BreastCancerDataset.csv').drop(['Unnamed: 32','id'], axis=1)
    feature_correlation_map(BreastCancerDataset)
    BreastCancerDataset_filtered = drop_highly_correlated_features(BreastCancerDataset.drop(['diagnosis'], axis=1), threshold=0.5)
    feature_correlation_map(BreastCancerDataset_filtered)
    
    dt2 = BreastCancerTreeTrainer(BreastCancerDataset_filtered.values, targets, classes)
    dt2.train()
    dt2.plot()
    print(f'The accuracy is: {dt2.accuracy()}')
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        

