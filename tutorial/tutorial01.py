'''
title: SMOTE for Imbalanced Classification with Python
reference: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
'''


from collections import Counter

from numpy import where, mean, std

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE #cria amostras no limite de decisão entre as duas classes
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score


class Tutorial:

    # create dataset
    def create_dataset(self):
        X, y = make_classification(n_samples=10_000, n_features=2, n_redundant=0, n_clusters_per_class=1,
                                   weights=[0.99],
                                   flip_y=0, random_state=1)
        return X, y


    # summarize class distribution
    def create_summarize(self, y):
        return Counter(y)


    # scatter plot of examples by class label
    def create_plot(self, X, y):
        counter = self.create_summarize(y)
        for label, _ in counter.items():
            row_ix = where(y == label)[0]
            plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
        plt.legend()
        plt.show()


    # create model decision tree
    def create_model(self, name):
        if name == 'decision_tree':
            model = DecisionTreeClassifier()
        return ('model', model)


    # oversampling
    def create_over_sampling(self, name, ss='auto', k_n=5):
        if name == 'smote':
            over = SMOTE(sampling_strategy=ss, k_neighbors=k_n)
        elif name == 'borderline_smote':
            over = BorderlineSMOTE(sampling_strategy=ss, k_neighbors=k_n)
        elif name == 'svm_smote':
            over = SVMSMOTE(sampling_strategy=ss, k_neighbors=k_n)
        elif name == 'adasyn':
            over = ADASYN(sampling_strategy=ss, n_neighbors=k_n)
        return ('over', over)


    # undersample random
    def create_under_sampling(self, name, ss='auto'):
        if name == 'ramdom_under_sampler':
            under = RandomUnderSampler(sampling_strategy=ss)
        return ('under', under)


    # create pipeline
    def create_pipeline(self, steps):
        pipeline = Pipeline(steps=steps)
        return pipeline


    # transform the dataset
    def transform_dataset(self, pipeline, X, y):
        X, y = pipeline.fit_resample(X, y)
        return X, y


    # evaluate
    def evaluate(self, pipeline, X, y, n_splits, n_repeats, scoring, k=5):
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
        scores = cross_val_score(pipeline, X, y, scoring=scoring, cv=cv, n_jobs=-1)
        print('k = {}, Mean {}: {:.2f}% ({:.2f}%)'.format(k, scoring, (mean(scores) * 100), (std(scores) * 100)))
