# SMOTE para classificação

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean, std


# define dataset
def define_dataset():
    X, y = make_classification(n_samples=10_000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=1)
    return X, y


# evaluate model
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=1)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('Mean ROC AUC: {:.2f}% ({:.2f}%)'.format((mean(scores)*100), (std(scores)*100)))


'''
MAIN
'''

# define dataset
X, y = define_dataset()

# define model
model = DecisionTreeClassifier()

# evaluate model
evaluate_model(model, X, y)  # Mean ROC AUC: 76.81% (8.02%)
