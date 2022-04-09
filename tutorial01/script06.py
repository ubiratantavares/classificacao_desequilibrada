# SMOTE para classificação - com subamostragem aleatória da classe majoritária

from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean, std


# create dataset
def create_dataset():
    X, y = make_classification(n_samples=10_000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=1)
    return X, y


# create pipeline
def create_pipeline():
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.9)
    model = DecisionTreeClassifier()
    steps = [('over', over), ('under', under), ('model', model)]
    pipeline = Pipeline(steps=steps)
    return pipeline


# evaluate
def evaluate(pipeline, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('Mean ROC AUC: {:.2f}% ({:.2f}%)'.format((mean(scores)*100), (std(scores)*100)))


'''
MAIN
'''

# create dataset
X, y = create_dataset()

# create pipeline
pipeline = create_pipeline()

# evaluate
evaluate(pipeline, X, y) # Mean ROC AUC: 84.83% (6.40%)