from tutoriais.tutorial01 import Tutorial

t = Tutorial()

'''
0.0 - Generate and plot a synthetic imbalanced classification dataset
'''
print('\n0.0 - Generate and plot a synthetic imbalanced classification dataset')

# create dataset
X_original, y_original = t.create_dataset()

# summarize class distribution
counter = t.summarize(y_original)
print(counter)

# scatter plot of examples by class label
t.scatter_plot(X_original, y_original)

'''
1.1 - Oversample and plot imbalanced dataset with SMOTE
'''
print('\n1.1 - Oversample and plot imbalanced dataset with SMOTE')

# oversample SMOTE
over = t.over_sample_smote()

# transform the dataset
X, y = t.transform_dataset(over, X_original, y_original)

# summarize class distribution
counter = t.summarize(y)
print(counter)

# scatter plot of examples by class label
t.scatter_plot(X, y)

'''
1.2 - Oversample and plot imbalanced dataset with borderline-SMOTE
'''
print('\n1.2 - Oversample and plot imbalanced dataset with borderline-SMOTE')

# oversample borderline SMOTE
over = t.over_sample_borderline_smote()

# transform the dataset
X, y = t.transform_dataset(over, X_original, y_original)

# summarize class distribution
counter = t.summarize(y)
print(counter)

# scatter plot of examples by class label
t.scatter_plot(X, y)


'''
1.3 - Oversample and plot imbalanced dataset with SVM bordeline-SMOTE
'''
print('\n1.3 - Oversample and plot imbalanced dataset with SVM bordeline-SMOTE')

# oversample SVM borderline SMOTE
over = t.over_sample_svm_borderline_smote()

# transform the dataset
X, y = t.transform_dataset(over, X_original, y_original)

# summarize class distribution
counter = t.summarize(y)
print(counter)

# scatter plot of examples by class label
t.scatter_plot(X, y)

'''
1.4 - Oversample and plot imbalanced dataset with ADASYN
'''
print('\n1.4 - Oversample and plot imbalanced dataset with ADASYN')

# oversample Adaptive Synthetic Sampling (ADASYN)
over = t.over_sample_adasyn()

# transform the dataset
X, y = t.transform_dataset(over, X_original, y_original)

# summarize class distribution
counter = t.summarize(y)
print(counter)

# scatter plot of examples by class label
t.scatter_plot(X, y)


'''
2.1 - Oversample with SMOTE and random undersample for imbalanced dataset
'''
print('\n2.1 - Oversample with SMOTE and random undersample for imbalanced dataset')

# create pipeline over_under
pipeline = t.pipeline_over_sample_smote_with_under()

# transform the dataset
X, y = t.transform_dataset(pipeline, X_original, y_original)

# summarize class distribution
counter = t.summarize(y)
print(counter)

# scatter plot of examples by class label
t.scatter_plot(X, y)


'''
2.2 - Oversample with borderline-SMOTE and random undersample for imbalanced dataset
'''
print('\n2.2 - Oversample with borderline-SMOTE and random undersample for imbalanced dataset')

# create pipeline over_under
pipeline = t.pipeline_over_sample_borderline_smote_with_under()

# transform the dataset
X, y = t.transform_dataset(pipeline, X_original, y_original)

# summarize class distribution
counter = t.summarize(y)
print(counter)

# scatter plot of examples by class label
t.scatter_plot(X, y)


'''
2.3 - Oversample with SVM borderline-SMOTE and random undersample for imbalanced dataset
'''
print('\n2.3 - Oversample with SVM borderline-SMOTE and random undersample for imbalanced dataset')

# create pipeline over_under
pipeline = t.pipeline_over_sample_svm_borderline_smote_with_under()

# transform the dataset
X, y = t.transform_dataset(pipeline, X_original, y_original)

# summarize class distribution
counter = t.summarize(y)
print(counter)

# scatter plot of examples by class label
t.scatter_plot(X, y)

'''
2.4 - Oversample with ADASYN and random undersample for imbalanced dataset
'''
print('\n2.4 - Oversample with ADASYN and random undersample for imbalanced dataset')

# create pipeline over_under
pipeline = t.pipeline_over_sample_adasyn_with_under()

# transform the dataset
X, y = t.transform_dataset(pipeline, X_original, y_original)

# summarize class distribution
counter = t.summarize(y)
print(counter)

# scatter plot of examples by class label
t.scatter_plot(X, y)


'''
3.1 - decision tree evaluated on imbalanced dataset
'''
print('\n3.1 - decision tree evaluated on imbalanced dataset')

# create model
model = t.model_decision_tree()

# evaluate model
t.evaluate(model, X_original, y_original)

'''
3.2 - decision tree evaluated on imbalanced dataset with SMOTE oversampling
'''
print('\n3.2 - decision tree evaluated on imbalanced dataset with SMOTE oversampling')

# create pipeline
pipeline = t.pipeline_over_sample_smote_and_decision_tree()

# evaluate pipeline
t.evaluate(pipeline, X_original, y_original)

'''
3.3 - decision tree evaluated on imbalanced dataset with borderline-SMOTE oversampling
'''
print('\n3.3 - decision tree evaluated on imbalanced dataset with borderline-SMOTE oversampling')

'''
3.4 - decision tree evaluated on imbalanced dataset with SVM borderline-SMOTE oversampling
'''
print('\n3.4 - decision tree evaluated on imbalanced dataset with SVM borderline-SMOTE oversampling')

'''
3.5 - decision tree evaluated on imbalanced dataset with ADASYN oversampling
'''
print('\n3.5 - decision tree evaluated on imbalanced dataset with ADASYN oversampling')

'''
4.1 - decision tree evaluated on imbalanced dataset with SMOTE oversampling and random undersampling
'''
print('\n4.1 - decision tree evaluated on imbalanced dataset with SMOTE oversampling and random undersampling')

# create pipeline
pipeline = t.pipeline_over_sample_smote_with_under_and_decision_tree()

# evaluate pipeline
t.evaluate(pipeline, X_original, y_original)


'''
4.2 - decision tree evaluated on imbalanced dataset with borderline-SMOTE oversampling and random undersampling
'''
print('\n4.2 - decision tree evaluated on imbalanced dataset with borderline-SMOTE oversampling and random undersampling')

# create pipeline
pipeline = t.pipeline_over_sample_borderline_smote_with_under_and_decision_tree()

# evaluate pipeline
t.evaluate(pipeline, X_original, y_original)


'''
4.3 - decision tree evaluated on imbalanced dataset with SVM borderline-SMOTE oversampling and random undersampling
'''
print('\n4.3 - decision tree evaluated on imbalanced dataset with SVM borderline-SMOTE oversampling and random undersampling')

# create pipeline
pipeline = t.pipeline_over_sample_svm_borderline_smote_with_under_and_decision_tree()

# evaluate pipeline
t.evaluate(pipeline, X_original, y_original)

'''
4.4 - decision tree evaluated on imbalanced dataset with ADASYN oversampling and random undersampling
'''
print('\n4.4 - decision tree evaluated on imbalanced dataset with ADASYN oversampling and random undersampling')

# create pipeline
pipeline = t.pipeline_over_sample_adasyn_with_under_and_decision_tree()

# evaluate pipeline
t.evaluate(pipeline, X_original, y_original)


'''
5.1 - Grid search k value for SMOTE oversampling for imbalanced classification
'''
print('\n5.1 - Grid search k value for SMOTE oversampling for imbalanced classification')

for k in range(1, 8):
    # create pipeline
    pipeline = t.pipeline_over_sample_smote_with_under_and_decision_tree_and_k(k)

    # evaluate model
    t.evaluate_k(pipeline, X_original, y_original, k)
