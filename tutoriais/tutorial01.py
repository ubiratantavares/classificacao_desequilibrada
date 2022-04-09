# referência: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

from collections import Counter

from numpy import where, mean, std

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE #cria apenas amostras no limite de decisão entre as duas classes
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
    def summarize(self, y):
        return Counter(y)

    # scatter plot of examples by class label
    def scatter_plot(self, X, y):
        counter = self.summarize(y)
        for label, _ in counter.items():
            row_ix = where(y == label)[0]
            plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
        plt.legend()
        plt.show()

    # create model decision tree
    def model_decision_tree(self):
        return DecisionTreeClassifier()

    # oversample SMOTE
    def over_sample_smote(self):
        return SMOTE()

    # oversample borderline SMOTE
    def over_sample_borderline_smote(self):
        return BorderlineSMOTE()

    # oversample SVM borderline SMOTE
    def over_sample_svm_borderline_smote(self):
        return SVMSMOTE()

    # oversample Adaptive Synthetic Sampling (ADASYN)
    def over_sample_adasyn(self):
        return ADASYN()

    # create pipeline: oversample SMOTE with under
    def pipeline_over_sample_smote_with_under(self):
        over = SMOTE(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.9)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        return pipeline


    # create pipeline: oversample borderline SMOTE with under
    def pipeline_over_sample_borderline_smote_with_under(self):
        over = BorderlineSMOTE(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.9)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        return pipeline


    # create pipeline: oversample SVM borderline SMOTE with under
    def pipeline_over_sample_svm_borderline_smote_with_under(self):
        over = SVMSMOTE(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.9)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        return pipeline

    # create pipeline: oversample ADASYN with under
    def pipeline_over_sample_adasyn_with_under(self):
        over = ADASYN(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.9)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        return pipeline


    # create pipeline: oversample SMOTE and decision tree
    def pipeline_over_sample_smote_and_decision_tree(self):
        over = SMOTE()
        model = DecisionTreeClassifier()
        steps = [('over', over), ('model', model)]
        pipeline = Pipeline(steps=steps)
        return pipeline

    # create pipeline: oversample SMOTE with under and decision tree
    def pipeline_over_sample_smote_with_under_and_decision_tree(self):
        over = SMOTE(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.9)
        model = DecisionTreeClassifier()
        steps = [('over', over), ('under', under), ('model', model)]
        pipeline = Pipeline(steps=steps)
        return pipeline

    # create pipeline: oversample borderline SMOTE with under and decision tree
    def pipeline_over_sample_borderline_smote_with_under_and_decision_tree(self):
        over = BorderlineSMOTE(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.9)
        model = DecisionTreeClassifier()
        steps = [('over', over), ('under', under), ('model', model)]
        pipeline = Pipeline(steps=steps)
        return pipeline


    # create pipeline: oversample SVM borderline SMOTE with under and decision tree
    def pipeline_over_sample_svm_borderline_smote_with_under_and_decision_tree(self):
        over = SVMSMOTE(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.9)
        model = DecisionTreeClassifier()
        steps = [('over', over), ('under', under), ('model', model)]
        pipeline = Pipeline(steps=steps)
        return pipeline


    # create pipeline: oversample ADASYN with under and decision tree
    def pipeline_over_sample_adasyn_with_under_and_decision_tree(self):
        over = ADASYN(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.9)
        model = DecisionTreeClassifier()
        steps = [('over', over), ('under', under), ('model', model)]
        pipeline = Pipeline(steps=steps)
        return pipeline


    # create pipeline: oversample SMOTE with under and decision tree and k_neighbors
    def pipeline_over_sample_smote_with_under_and_decision_tree_and_k(self, k):
        over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
        under = RandomUnderSampler(sampling_strategy=0.9)
        model = DecisionTreeClassifier()
        steps = [('over', over), ('under', under), ('model', model)]
        pipeline = Pipeline(steps=steps)
        return pipeline


    # transform the dataset
    def transform_dataset(self, model, X, y):
        X, y = model.fit_resample(X, y)
        return X, y


    # evaluate model
    def evaluate(self, model, X, y):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=1)
        scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
        print('Mean ROC AUC: {:.2f}% ({:.2f}%)'.format((mean(scores) * 100), (std(scores) * 100)))

    # evaluate model for k_neighbors
    def evaluate_k(self, model, X, y, k):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=1)
        scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
        print('k = {}, Mean ROC AUC: {:.2f}% ({:.2f}%)'.format(k, (mean(scores) * 100), (std(scores) * 100)))