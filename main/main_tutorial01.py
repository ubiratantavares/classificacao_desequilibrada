from tutorial.tutorial01 import Tutorial

t = Tutorial()

'''
Generate and plot a synthetic imbalanced classification dataset
'''
print('\nGenerate and plot a synthetic imbalanced classification dataset')

# create dataset
X_original, y_original = t.create_dataset()

# summarize class distribution
counter = t.create_summarize(y_original)
print(counter)

# scatter plot of examples by class label
t.create_plot(X_original, y_original)

methods = ['smote', 'borderline_smote', 'svm_smote', 'adasyn']

'''
Oversample and plot imbalanced dataset with method
'''
for o in methods:
    print('\nOversample and plot imbalanced dataset with {}'.format(o))

    # over-sampling
    over = t.create_over_sampling(o)

    # create pipeline
    steps = [over]
    pipeline = t.create_pipeline(steps)

    # transform the dataset
    X, y = t.transform_dataset(pipeline, X_original, y_original)

    # summarize
    counter = t.create_summarize(y)
    print(counter)

    # scatter plot of examples by class label
    t.create_plot(X, y)

'''
Oversample with SMOTE and random undersample for imbalanced dataset
'''
for o in methods:
    print('\nOversample with {} and random undersample for imbalanced dataset'.format(o))

    # over-sampling
    over = t.create_over_sampling(o, 0.1)

    # under-sampling
    under = t.create_under_sampling('ramdom_under_sampler', 0.9)

    # create pipeline
    steps = [over, under]
    pipeline = t.create_pipeline(steps)

    # transform the dataset
    X, y = t.transform_dataset(pipeline, X_original, y_original)

    # summarize
    counter = t.create_summarize(y)
    print(counter)

    # scatter plot of examples by class label
    t.create_plot(X, y)

'''
Decision tree evaluated on imbalanced dataset
'''
print('\nDecision tree evaluated on imbalanced dataset')

# create model
model = t.create_model('decision_tree')

# create pipeline
steps = [model]
pipeline = t.create_pipeline(steps)

# evaluate model
t.evaluate(pipeline, X_original, y_original, 10, 30, 'roc_auc')

'''
Decision tree evaluated on imbalanced dataset with method oversampling
'''
for o in methods:
    print('\nDecision tree evaluated on imbalanced dataset with {} oversampling'.format(o))

    # over-sampling
    over = t.create_over_sampling(o)

    # create model
    model = t.create_model('decision_tree')

    # create pipeline
    steps = [over, model]
    pipeline = t.create_pipeline(steps)

    # evaluate model
    t.evaluate(pipeline, X_original, y_original, 10, 30, 'roc_auc')


'''
Decision tree evaluated on imbalanced dataset with method oversampling and random undersampling
'''
for o in methods:
    print('\nDecision tree evaluated on imbalanced dataset with {} oversampling and random undersampling'.format(o))

    # over-sampling
    over = t.create_over_sampling(o, 0.1)

    # under-sampling
    under = t.create_under_sampling('ramdom_under_sampler', 0.9)

    # create model
    model = t.create_model('decision_tree')

    # create pipeline
    steps = [over, under, model]
    pipeline = t.create_pipeline(steps)

    # evaluate model
    t.evaluate(pipeline, X_original, y_original, 10, 30, 'roc_auc')

'''
Grid search k value for method oversampling and random undersampling for imbalanced classification
'''
for o in methods:
    print('\nGrid search k value for {} oversampling and random undersampling for imbalanced classification'.format(o))
    for k in range(1, 8):
        # over-sampling
        over = t.create_over_sampling(o, 0.1, k)

        # under-sampling
        under = t.create_under_sampling('ramdom_under_sampler', 0.9)

        # create model
        model = t.create_model('decision_tree')

        # create pipeline
        steps = [over, under, model]
        pipeline = t.create_pipeline(steps)

        # evaluate model
        t.evaluate(pipeline, X_original, y_original, 10, 30, 'roc_auc', k)
