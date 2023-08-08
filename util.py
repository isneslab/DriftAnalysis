import numpy as np
import os
import pickle
import string
import random

from sklearn import svm
from sklearn.model_selection import train_test_split

def feature_reduction(X, y, feature_names, file_path, feature_size=1000):
    """Helper function for feature reduction 

    Args:
        X (np.ndarray): Array of predictors X
        y (np.ndarray): Array of predictors y
        feature_names (np.ndarray): Array of feature names
        file_path (str): File path to save feature reduced indexes
        feature_size (int, optional): Feature reduction size. Defaults to 1000.

    Returns:
        (np.ndarray, np.ndarray)
            Array of reduced feature predictors X
            Array of reduced feature names
    """    
    print("Reducing feature space...")

    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            select_index = pickle.load(file)
    else:
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=3)
        svm_ = svm.LinearSVC(max_iter=50000, random_state=3)
        svm_.fit(X_train, y_train)
        coef = svm_.coef_
        select_index = np.argpartition(abs(coef[0]), -feature_size)[-feature_size:]
        with open(file_path, "wb") as file:
            pickle.dump(select_index,file)

    X = X[:,select_index].toarray()
    feature_names = feature_names[select_index]
    return X, feature_names


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str