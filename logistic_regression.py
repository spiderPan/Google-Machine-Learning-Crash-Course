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

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))


def preprocess_features(california_housing_dataframe):
    selected_features = california_housing_dataframe[["latitude",
                                                      "longitude",
                                                      "housing_median_age",
                                                      "total_rooms",
                                                      "total_bedrooms",
                                                      "population",
                                                      "households",
                                                      "median_income"]]
    preprocess_features = selected_features.copy()
    display.display(selected_features.dtypes)
    preprocess_features['rooms_per_person'] = california_housing_dataframe['total_rooms'] / california_housing_dataframe['population']

    return preprocess_features


def preprocess_targets(california_housing_dataframe):
    output_targets = pd.DataFrame()

    output_targets['median_house_value_is_high'] = (california_housing_dataframe['median_house_value'] > 265000).astype(float)
    return output_targets


def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(100000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_linear_regressor_model(learning_rate,
                                 steps,
                                 batch_size,
                                 training_examples,
                                 training_targets,
                                 validation_examples,
                                 validation_targets):
    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    def training_input_fn(): return my_input_fn(training_examples, training_targets['median_house_value_is_high'], batch_size=batch_size)

    def predict_training_input_fn(): return my_input_fn(training_examples, training_targets['median_house_value_is_high'], num_epochs=1, shuffle=False)

    def predict_validation_input_fn(): return my_input_fn(validation_examples, validation_targets['median_house_value_is_high'], num_epochs=1, shuffle=False)

    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute predictions.
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))

        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print ("Model training finished.")

    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title('RMSE vs Periods')
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()

    return linear_regressor


def train_linear_classifier_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.esitmator.clip_gradients_by_norm(my_optimizer, 5.0)

    linear_classifier = tf.estimator.LinearClassifier(feature_columns=construct_feature_columns(training_examples), optimizer=my_optimizer)

    def training_input_fn(): return my_input_fn(training_examples, training_targets['median_house_value_is_high'], batch_size=batch_size)

    def predict_training_input_fn(): return my_input_fn(training_examples, training_targets['median_house_value_is_high'], num_epochs=1, shuffle=False)

    def predict_validation_input_fn(): return my_input_fn(validation_examples, validation_targets['median_house_value_is_high'], num_epochs=1, shuffle=False)

    print('Training model...')
    print("RMSE (on training data):")
    training_log_loss = []
    validation_log_loss = []

    for period in range(0, periods):
        linear_classifier.train(input_fn=training_input_fn, steps=steps_per_period)
        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
        validation_probabilites = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilites = np.array([item['probabilities'] for item in validation_probabilites])
        # Computer LogLoss
        training_log_loss = metrics.log_loss(training_targets, training_predictions)
        validation_log_loss = metrics.log_loss(validation_targets, validation_predictions)

        print(" period %02d : %0.2f") % (period, training_root_mean_squared_error)
        training_log_loss.append(training_log_loss)
        validation_log_loss.append(validation_log_loss)
    print('Model trianing finished.')

    plt.ylabel('LogLoss')
    plt.xlabel('Periods')
    plt.title('LogLoss vs. Periods')
    plt.tight_layout()
    plt.plot(training_log_loss, label('training'))
    plt.plot(validation_log_loss, label('validation'))
    plt.legent()


training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

# Double-check that we've done the right thing.
print ("Training examples summary:")
display.display(training_examples.describe())
print ("Validation examples summary:")
display.display(validation_examples.describe())

print ("Training targets summary:")
display.display(training_targets.describe())
print ("Validation targets summary:")
display.display(validation_targets.describe())

linear_regressor = train_linear_regressor_model(
    learning_rate=0.000001,
    steps=200,
    batch_size=20,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

linear_classifier = train_linear_classifier_model(
    learning_rate=0.000005,
    steps=500,
    batch_size=20,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets
)
