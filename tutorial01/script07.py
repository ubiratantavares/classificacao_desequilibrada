# SMOTE para classificação - com subamostragem aleatória da classe majoritária
# teste de diferentes valores de k vizinhos mais próximos selecionados para o SMOTE (padrão é k = 5)

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
def create_pipeline(k):
    over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
    under = RandomUnderSampler(sampling_strategy=0.9)
    model = DecisionTreeClassifier()
    steps = [('over', over), ('under', under), ('model', model)]
    pipeline = Pipeline(steps=steps)
    return pipeline


# evaluate
def evaluate(pipeline, X, y, k):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('k = {}, Mean ROC AUC: {:.2f}% ({:.2f}%)'.format(k, (mean(scores)*100), (std(scores)*100)))


'''
MAIN
'''

# create dataset
X, y = create_dataset()

for k in range(1, 8):
    # create pipeline
    pipeline = create_pipeline(k)
    # evaluate
    evaluate(pipeline, X, y, k)

'''
k = 1, Mean ROC AUC: 83.63% (6.99%)
k = 2, Mean ROC AUC: 83.85% (6.48%)
k = 3, Mean ROC AUC: 84.09% (6.61%)
k = 4, Mean ROC AUC: 84.64% (6.67%)
k = 5, Mean ROC AUC: 84.48% (6.53%)
k = 6, Mean ROC AUC: 84.65% (6.18%)
k = 7, Mean ROC AUC: 84.58% (6.33%)
'''
