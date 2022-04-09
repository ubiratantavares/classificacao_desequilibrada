# borderline-SMOTE for imbalanced dataset

from sklearn.datasets import make_classification
from collections import Counter
from numpy import where
import matplotlib.pyplot as plt

# cria apenas exemplos sintéticos ao longo do limite de decisão entre as duas classes
from imblearn.over_sampling import BorderlineSMOTE

# create dataset
def create_dataset():
    X, y = make_classification(n_samples=10_000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=1)
    return X, y


# summarize class distribution
def summarize(y):
    return Counter(y)


# scatter plot of examples by class label
def scatter_plot(X, y):
    counter = summarize(y)
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    plt.legend()
    plt.show()


# transform the dataset
def transform_dataset(X, y):
    oversample = BorderlineSMOTE()
    X, y = oversample.fit_resample(X, y)
    return X, y

'''
MAIN
'''

# define dataset
X, y = create_dataset()

# summarize class distribution
print(summarize(y))

# scatter plot of examples by class label
scatter_plot(X, y)

# transform the dataset
X, y = transform_dataset(X, y)

# summarize class distribution
print(summarize(y))

# scatter plot of examples by class label
scatter_plot(X, y)
