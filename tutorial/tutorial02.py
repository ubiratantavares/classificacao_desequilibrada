'''
title: Multi-Class Imbalanced Classification
reference: https://machinelearningmastery.com/multi-class-imbalanced-classification/
'''
import pandas as pd
from numpy import where, mean, std

from pandas import read_csv
from collections import Counter
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE #cria amostras no limite de decisão entre as duas classes
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

class Tutorial:

    # load the csv file as a data frame
    def load_csv(self, path_filename):
        df = read_csv(path_filename, header=None)
        data = df.values
        return data

    # split into input and output elements
    def split_input_output(self, data):
        X, y = data[:, :-1], data[:, -1]
        return X, y

    # label encode the target variable
    def label_encode(self, y):
        y_transform = LabelEncoder().fit_transform(y)
        return y_transform

    # summarize class distribution
    def create_summarize(self, y):
        counter = Counter(y)
        return counter

    # print summarize
    def print_summarize(self, y, counter):
        for k, v in counter.items():
            per = v / len(y) * 100
            print('Class={}, n={} ({:.3f}%)'.format(k, v, per))

    # plot the distribution
    def create_plot_bar(self, counter):
        '''
        plt.bar(counter.keys(), counter.values())
        '''
        df = pd.DataFrame(counter.items(), columns=['class', 'frequency'])
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x='class', y="frequency", data=df)
        plt.show()

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

    # create pipeline
    def create_pipeline(self, steps):
        pipeline = Pipeline(steps=steps)
        return pipeline


    # transform the dataset
    def transform_dataset(self, pipeline, X, y):
        X, y = pipeline.fit_resample(X, y)
        return X, y


    # evaluate
    def evaluate(self, pipeline, X, y, parameters):
        n_splits, n_repeats, scoring = parameters
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
        scores = cross_val_score(pipeline, X, y, scoring=scoring, cv=cv, n_jobs=-1)
        print('Mean {}: {:.2f}% ({:.2f}%)'.format(scoring, (mean(scores) * 100), (std(scores) * 100)))


    # create model
    def create_model(self, name, parameters):
        if name == 'random_forest_classifier':
            n_estimators, class_weight = parameters
            model = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight)
        return ('model', model)