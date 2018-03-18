import math

import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt
from sklearn import metrics
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv('https://storage.googleapis.com/mledu-datasets/california_housing_train.csv', sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
# display.display(california_housing_dataframe.describe())


def preprocess_features(california_housing_dataframe):
    selected_features = california_housing_dataframe[['latitude',
                                                      'longitude',
                                                      'housing_median_age',
                                                      'total_rooms',
                                                      'total_bedrooms',
                                                      'population',
                                                      'households',
                                                      'median_income']]
    preprocess_features = selected_features.copy()
    preprocess_features['rooms_perperson'] = preprocess_features['total_rooms'] / preprocess_features['population']
    return preprocess_features


def preprocess_targets(california_housing_dataframe):
    output_targets = pd.DataFrame()
    output_targets['median_house_value'] = california_housing_dataframe['median_house_value'] / 1000.0
    return output_targets


def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=False, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = tf.data.Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def select_and_transform_features(source_df):
    Latitude_Ranges = zip(range(32, 44), range(33, 45))
    selected_examples = pd.DataFrame()
    selected_examples['median_income'] = source_df['median_income']
    for r in Latitude_Ranges:
        selected_examples["latitude_%d_to_%d" % (r[0], r[1])] = source_df['latitude'].apply(lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)

    return selected_examples

def train_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):
    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=construct_feature_columns(training_examples), optimizer=my_optimizer)

    training_input_fn = lambda: my_input_fn(training_examples,training_targets['median_house_value'],batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,training_targets['median_house_value'],num_epochs=1,shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets['median_house_value'],num_epochs=1,shuffle=False)

    print ('Training model...')
    print ('RMSE (on training data):')
    training_rmse = []
    validation_rmse = []

    for period in range(0, periods):
        linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict(input_fn = predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        training_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(training_predictions,training_targets))
        validation_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(validation_predictions,validation_targets))
        print (' period %02d : %0.2f' %(period, training_root_mean_squared_error))
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print ('Model training finished')



training_dataframe = california_housing_dataframe.head(12000)
validation_dataframe = california_housing_dataframe.tail(5000)
training_examples = preprocess_features(training_dataframe)
training_targets = preprocess_targets(training_dataframe)
validation_examples = preprocess_features(validation_dataframe)
validation_targets = preprocess_targets(validation_dataframe)

print('Training examples summary:')
display.display(training_examples.describe())
print('Validation examples summary:')
display.display(validation_examples.describe())

print('Training targets summary:')
display.display(training_targets.describe())
print('Validation targets summary:')
display.display(validation_targets.describe())

correlation_dataframe = training_examples.copy()
correlation_dataframe['target'] = training_targets['median_house_value']

display.display(correlation_dataframe.corr())
plt.scatter(training_examples["latitude"], training_targets["median_house_value"])

selected_training_examples = select_and_transform_features(training_examples)
selected_validation_examples = select_and_transform_features(validation_examples)

minimal_features = [
    'median_income',
]


assert minimal_features, "You must select at least one feature!"

minimal_training_examples = training_examples[minimal_features]
minimal_validation_examples = validation_examples[minimal_features]

train_model(
    learning_rate=0.01,
    steps=500,
    batch_size=5,
    training_examples=selected_training_examples,
    training_targets=training_targets,
    validation_examples=selected_validation_examples,
    validation_targets=validation_targets,
)
