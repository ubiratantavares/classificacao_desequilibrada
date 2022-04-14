from tutorial.tutorial02 import Tutorial

import warnings
warnings.filterwarnings('ignore')

t = Tutorial()

'''
SMOTE Oversampling for Multi-Class Classification
'''
print('\nload and summarize the dataset')

# load the csv file as a data frame
data = t.load_csv("../dataset/glass.csv")

# split into input and output elements
X, y = t.split_input_output(data)

# label encode the target variable
y_transform = t.label_encode(y)

# summarize distribution
counter = t.create_summarize(y_transform)
print(counter)


# print summarize
t.print_summarize(y, counter)

# plot the distribution
t.create_plot_bar(counter)

'''
example of oversampling a multi-class classification dataset
'''
print('\nexample of oversampling a multi-class classification dataset')

# transform the dataset
oversample = t.create_over_sampling('smote')

# create pipeline
steps = [oversample]
pipeline = t.create_pipeline(steps)

X_over, y_over = t.transform_dataset(pipeline, X, y_transform)

# summarize distribution
counter = t.create_summarize(y_over)

# print summarize
t.print_summarize(y_over, counter)

# plot the distribution
t.create_plot_bar(counter)


'''
example of oversampling a multi-class classification dataset with a custom strategy
'''
print('example of oversampling a multi-class classification dataset with a custom strategy')

# transform the dataset
strategy = {0: 100, 1: 100, 2: 200, 3: 200, 4: 200, 5: 200}

# transform the dataset
oversample = t.create_over_sampling('smote', ss=strategy)

# create pipeline
steps = [oversample]
pipeline = t.create_pipeline(steps)

X_over, y_over = t.transform_dataset(pipeline, X, y_transform)

# summarize distribution
counter = t.create_summarize(y_over)

# print summarize
t.print_summarize(y_over, counter)

# plot the distribution
t.create_plot_bar(counter)

'''
baseline model and test harness for the glass identification dataset
'''
print('\nbaseline model and test harness for the glass identification dataset')

# create model
parameters = (1000, None)
model = t.create_model('random_forest_classifier', parameters)

# create pipeline
steps = [model]
pipeline = t.create_pipeline(steps)

# evaluate model
parameters = (10, 30, 'accuracy')
t.evaluate(pipeline, X, y_transform, parameters)

'''
cost sensitive random forest with default class weights
'''
print('\ncost sensitive random forest with default class weights')

# create model
parameters = (1000, 'balanced')
model = t.create_model('random_forest_classifier', parameters)

# create pipeline
steps = [model]
pipeline = t.create_pipeline(steps)

# evaluate model
parameters = (10, 30, 'accuracy')
t.evaluate(pipeline, X, y_transform, parameters)

'''
cost sensitive random forest with custom class weightings
'''
print('\ncost sensitive random forest with custom class weightings')

weights = {0: 1.0, 1: 1.0, 2: 2.0,  3: 2.0,  4: 2.0, 5: 2.0}

# create model
parameters = (1000, weights)
model = t.create_model('random_forest_classifier', parameters)

# create pipeline
steps = [model]
pipeline = t.create_pipeline(steps)

# evaluate model
parameters = (10, 30, 'accuracy')
t.evaluate(pipeline, X, y_transform, parameters)
