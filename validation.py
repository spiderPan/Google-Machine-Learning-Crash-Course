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

california_housing_dataframe = pd.read_csv(
    "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")


def preprocess_features(california_housing_dataframe):
    selected_features = california_housing_dataframe[
        ['latitude',
         'longitude',
         'housing_median_age',
         'total_rooms',
         'total_bedrooms',
         'population',
         'households',
         'median_income']
    ]
    preprocess_features = selected_features.copy()

    preprocess_features['room_per_person'] = california_housing_dataframe[
        'total_rooms'] / california_housing_dataframe['population']
    return preprocess_features


def preprocess_targets(california_housing_dataframe):
    output_features = pd.DataFrame()
    output_features['median_house_value'] = california_housing_dataframe[
        'median_house_value'] / 1000.0
    return output_features


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = DataFrame.from_tesnor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(100000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def construct_feature_columns(input_features):
    return set([tr.feature_column.numeric_column(my_feature) for my_feature in input_features])


def train_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):
    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(
        my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    def training_input_fn(): return my_input_fn(
        training_examples, training_targets['median_house_value'], batch_size=batch_size)

    def predict_training_input_fn(): return my_input_fn(
        training_examples, training_targets['median_house_value'], num_epochs=1, shuffle=False)

    def predict_validation_input_fn(): return my_input_fn(
        validation_examples, validation_targets['median_house_value'], num_epochs=1, shuffle=False)

    print('Training model...')
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []

    for period in range(0, periods):
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period)

        training_predictions = linear_regressor.predict(
            input_fn=predict_training_input_fn)
        training_predictions = np.array(
            [item['predictions'][0] for item in training_predictions])
        validation_predictions = linear_regressor.predict(
            input_fn=predict_validation_input_fn)
        validation_predictions = np.array(
            [item['predictions'][0] for item in validation_predictions])

        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))

        print(" period %02d : %0.2f") % (
            period, training_root_mean_squared_error)
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print('Model trianing finished.')

    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title('Root Mean Squared Error vs. Periods')
    plt.tight_layout()
    plt.plot(trianing_rmse, label('training'))
    plt.plot(validation_rmse, label('validation'))
    plt.legent()

    return linear_regressor


randomlize_data = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
training_dataframe = randomlize_data.head(12000)
validation_dataframe = randomlize_data.tail(5000)
training_examples = preprocess_features(training_dataframe)
# training_examples.describe()
training_targets = preprocess_targets(training_dataframe)
# training_targets.describe()
validation_examples = preprocess_features(validation_dataframe)
# validation_examples.describe()
validation_targets = preprocess_targets(validation_dataframe)
# validation_targets.describe()

plt.figure(figsize=(13, 8))
ax = plt.subplot(1, 2, 1)
ax.set_title('Validation Data')
ax.set_autoscalex_on(False)
ax.set_ylim([32, 43])
ax.set_xlim([-126, -112])
plt.scatter(validation_examples['longitude'],
            validation_examples['latitude'],
            cmap='coolwarm',
            c=validation_targets['median_house_value'] / validation_targets['median_house_value'].max())

ax = plt.subplot(1, 2, 2)
ax.set_title('Training Data')
ax.set_autoscalex_on(False)
ax.set_ylim([32, 43])
ax.set_xlim([-126, -112])
plt.scatter(training_examples['longitude'],
            training_examples['latitude'],
            cmap='coolwarm',
            c=training_targets['median_house_value'] / training_targets['median_house_value'].max())

_ = plt.plot()

california_housing_test_data = pd.read_csv(
    "https://storage.googleapis.com/mledu-datasets/california_housing_test.csv", sep=",")
test_example = preprocess_features(california_housing_test_data)
test_targets = preprocess_targets(california_housing_test_data)


def predict_test_input_fn(): return my_input_fn(
    test_example,
    test_targets['median_house_value'],
    num_epochs=1,
    shuffle=False
)


test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item['predictions'][0]
                             for item in test_predictions])

root_mean_squared_error = math.sqrt(
    matrics.mean_squared_error(test_predictions, test_targets))

print("Final MRSE on test data : %0.2f" % root_mean_squared_error)
