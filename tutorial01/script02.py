# SMOTE para balanceamento de dados

import imblearn
print(imblearn.__version__)

from sklearn.datasets import make_classification
from collections import Counter
from numpy import where
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


# define dataset
def define_dataset():
    X, y = make_classification(n_samples=10_000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=1)
    return X, y


# summarize class distribution
def summarize_class_distribution(y):
    return Counter(y)


# scatter plot of examples by class label
def scatter_plot(X, y):
    counter = summarize_class_distribution(y)
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    plt.legend()
    plt.show()


# transform the dataset
def transform_dataset(X, y):
    over = SMOTE()
    X, y = over.fit_resample(X, y)
    return X, y

'''
MAIN
'''

# define dataset
X, y =  define_dataset()

# summarize class distribution
print(summarize_class_distribution(y))

# scatter plot of examples by class label
scatter_plot(X, y)

# transform the dataset
X, y = transform_dataset(X, y)

# summarize class distribution
print(summarize_class_distribution(y))

# scatter plot of examples by class label
scatter_plot(X, y)
