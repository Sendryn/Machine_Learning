# Author: Sandrine Neang
# Date: 13/10/2023
# Project: 08 boosting
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, RandomizedSearchCV)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, precision_score)

from tools import get_titanic, build_kaggle_submission


def get_better_titanic():
    '''
    Loads the cleaned titanic dataset but change
    how we handle the age column.
    '''
    # Load in the raw data
    # check if data directory exists for Mimir submissions
    # DO NOT REMOVE
    if os.path.exists('./data/train.csv'):
        train = pd.read_csv('./data/train.csv')
        test = pd.read_csv('./data/test.csv')
    else:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')

    # Concatenate the train and test set into a single dataframe
    # we drop the `Survived` column from the train set
    X_full = pd.concat([train.drop('Survived', axis=1), test], axis=0)

    # The cabin category consist of a letter and a number.
    # We can divide the cabin category by extracting the first
    # letter and use that to create a new category. So before we
    # drop the `Cabin` column we extract these values
    X_full['Cabin_mapped'] = X_full['Cabin'].astype(str).str[0]
    # Then we transform the letters into numbers
    cabin_dict = {k: i for i, k in enumerate(X_full.Cabin_mapped.unique())}
    X_full.loc[:, 'Cabin_mapped'] =\
        X_full.loc[:, 'Cabin_mapped'].map(cabin_dict)

    # We drop multiple columns that contain a lot of NaN values
    # in this assignment
    # Maybe we should
    X_full.drop(
        ['PassengerId', 'Cabin', 'Name', 'Ticket'],
        inplace=True, axis=1)
    
    age_mean = X_full[X_full.Age != np.nan].Age.mean()
    X_full['Age'].fillna(age_mean, inplace=True)

    # Instead of dropping the fare column we replace NaN values
    # with the 3rd class passenger fare mean.
    fare_mean = X_full[X_full.Pclass == 3].Fare.mean()
    X_full['Fare'].fillna(fare_mean, inplace=True)
    # Instead of dropping the Embarked column we replace NaN values
    # with `S` denoting Southampton, the most common embarking
    # location
    X_full['Embarked'].fillna('S', inplace=True)

    # We then use the get_dummies function to transform text
    # and non-numerical values into binary categories.
    X_dummies = pd.get_dummies(
        X_full,
        columns=['Sex', 'Cabin_mapped', 'Embarked'],
        drop_first=True)

    # We now have the cleaned data we can use in the assignment
    X = X_dummies[:len(train)]
    submission_X = X_dummies[len(train):]
    y = train.Survived
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=5, stratify=y)
    
    return (X_train, y_train), (X_test, y_test), submission_X



def rfc_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a random forest classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''
    # Initialize the Random Forest Classifier
    rfc = RandomForestClassifier()

    # Train the classifier on the training set
    rfc.fit(X_train, t_train)

    # Make predictions on the test set
    predictions = rfc.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(t_test, predictions)
    precision = precision_score(t_test, predictions)
    recall = recall_score(t_test, predictions)

    return accuracy, precision, recall


def gb_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a Gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''
    # Initialize the GradientBoostingClassifier
    gb = GradientBoostingClassifier()

    # Train the classifier on the training set
    gb.fit(X_train, t_train)

    # Make predictions on the test set
    predictions = gb.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(t_test, predictions)
    precision = precision_score(t_test, predictions)
    recall = recall_score(t_test, predictions)

    return accuracy, precision, recall


def param_search(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    # Create the parameter grid
    gb_param_grid = {
        'n_estimators': [i for i in range(0,100)],
        'max_depth': [i for i in range(0,50)],
        'learning_rate': [0.05*i for i in range(20)]}
    # Instantiate the regressor
    gb = GradientBoostingClassifier()
    # Perform random search
    gb_random = RandomizedSearchCV(
        param_distributions=gb_param_grid,
        estimator=gb,
        scoring="accuracy",
        verbose=0,
        n_iter=50,
        cv=4)
    # Fit randomized_mse to the data
    gb_random.fit(X, y)
    # Print the best parameters and lowest RMSE
    return gb_random.best_params_


def gb_optimized_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test) with
    your own optimized parameters
    '''
    gb = GradientBoostingClassifier(learning_rate=0.35, n_estimators=35, max_depth=3)
    gb.fit(X_train, t_train)
    predictions = gb.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(t_test, predictions)
    precision = precision_score(t_test, predictions)
    recall = recall_score(t_test, predictions)

    return accuracy, precision, recall


def _create_submission():
    '''Create your kaggle submission
    '''
    pass
    prediction = None # !!! Your prediction here !!!
    build_kaggle_submission(prediction)


#(tr_X, tr_y), (tst_X, tst_y), submission_X = get_titanic()
#rfc_train_test(tr_X, tr_y, tst_X, tst_y)
#gb_train_test(tr_X, tr_y, tst_X, tst_y)
#param_search(tr_X, tr_y)
#gb_optimized_train_test(tr_X, tr_y, tst_X, tst_y)
