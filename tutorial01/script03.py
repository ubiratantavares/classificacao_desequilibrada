# SMOTE para balanceamento de dados - com subamostragem aleatória da classe majoritária

from sklearn.datasets import make_classification
from collections import Counter
from numpy import where
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


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


# define pipeline
def define_pipeline():
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.9)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    return pipeline


# transform the dataset
def transform_dataset(pipeline, X, y):
    return pipeline.fit_resample(X, y)


'''
MAIN
'''

# define dataset
X, y = define_dataset()

# summarize class distribution
print(summarize_class_distribution(y))

# scatter plot of examples by class label
scatter_plot(X, y)

# define pipeline
pipeline = define_pipeline()

# transform the dataset
X, y = transform_dataset(pipeline, X, y)

# summarize class distribution
print(summarize_class_distribution(y))

# scatter plot of examples by class label
scatter_plot(X, y)
