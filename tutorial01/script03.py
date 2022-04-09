# SMOTE para balanceamento de dados - com subamostragem aleatória da classe majoritária

from sklearn.datasets import make_classification
from collections import Counter
from numpy import where
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# define dataset
X, y = make_classification(n_samples=10_000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                           flip_y=0, random_state=1)

# summarize class distribution
counter = Counter(y)
print(counter)

# transform the dataset
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.9)

# define pipeline
steps = [('o', over), ('u', under)]

pipeline = Pipeline(steps=steps)

# transform the dataset
X, y = pipeline.fit_resample(X, y)

# summarize the new class distribution
counter = Counter(y)
print(counter)

# scatter plot of examples by class label
for label, _ in counter.items():
    row_ix = where(y == label)[0]
    plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
plt.legend()
plt.show()
