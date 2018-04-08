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


def preprocess_features(california_housing_dataframe):
    selected_features = california_housing_dataframe[
        ["latitude",
         "longitude",
         "housing_median_age",
         "total_rooms",
         "total_bedrooms",
         "population",
         "households",
         "median_income"
         ]]
    processed_features = selected_features.copy()
    processed_features["rooms_per_person"] = processed_features["total_rooms"] / processed_features["population"]

    return processed_features


def preprocess_targets(california_housing_dataframe):
    output_targets = pd.DataFrame()
    output_targets["median_house_value"] = california_housing_dataframe["median_house_value"] / 1000.0
    return output_targets


def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])


def linear_scale(series):
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / 2.0
    return series.apply(lambda x: ((x - min_val) / scale) - 1.0)


def normalize_linear_scale(examples_dataframe):
    normalized_dataframe = pd.DataFrame()
    for col in examples_dataframe:
        normalized_dataframe[col] = linear_scale(examples_dataframe[col])
    return normalized_dataframe


def log_normalize(series):
    return series.apply(lambda x: math.log(x + 1.0))


def clip(series, clip_to_min, clip_to_max):
    return series.apply(lambda x: (min(max(x, clip_to_min), clip_to_max)))


def binary_threshold(series, threshold):
    return series.apply(lambda x: (1 if x > threshold else 0))


def normalize(example_data_frame):
    normalized_dataframe = pd.DataFrame()
    normalized_dataframe['households'] = log_normalize(example_data_frame['households'])
    normalized_dataframe['median_income'] = log_normalize(example_data_frame['median_income'])
    normalized_dataframe['total_bedrooms'] = log_normalize(example_data_frame['total_bedrooms'])

    normalized_dataframe['housing_median_age'] = linear_scale(example_data_frame['housing_median_age'])
    normalized_dataframe['latitude'] = linear_scale(example_data_frame['latitude'])
    normalized_dataframe['longitude'] = linear_scale(example_data_frame['longitude'])

    normalized_dataframe['population'] = linear_scale(clip(example_data_frame['population'], 0, 5000))
    normalized_dataframe['rooms_per_person'] = linear_scale(clip(example_data_frame['rooms_per_person'], 0, 5))
    normalized_dataframe['total_rooms'] = linear_scale(clip(example_data_frame['total_rooms'], 0, 10000))

    return normalized_dataframe


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_nn_regression_model(my_optimizer, steps, batch_size, hidden_units, training_examples, training_targets, validation_examples, validation_targets):
    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    dnn_regressor = tf.estimator.DNNRegressor(feature_columns=construct_feature_columns(training_examples), hidden_units=hidden_units, optimizer=my_optimizer)

    def training_input_fn(): return my_input_fn(training_examples, training_targets['median_house_value'], batch_size=batch_size)

    def predict_training_input_fn(): return my_input_fn(training_examples, training_targets['median_house_value'], shuffle=False, num_epochs=1)

    def predict_validation_input_fn(): return my_input_fn(validation_examples, validation_targets['median_house_value'], shuffle=False, num_epochs=1)

    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []

    for period in range(0, periods):
        dnn_regressor.train(input_fn=training_input_fn, steps=steps_per_period)

        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        training_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(validation_predictions, validation_targets))

        print(" period %02d : %0.2f" % (period, training_root_mean_squared_error))
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")

    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("RMSE vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()

    print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

    return dnn_regressor, training_rmse, validation_rmse


california_housing_dataframe = pd.read_csv('data/california_housing_train.csv', sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
normalized_dataframe = normalize_linear_scale(preprocess_features(california_housing_dataframe))
normalized_training_examples = normalized_dataframe.head(12000)
normalized_validation_examples = normalized_dataframe.tail(5000)

training_dataframe = california_housing_dataframe.head(12000)
validation_dataframe = california_housing_dataframe.tail(50000)

training_examples = preprocess_features(training_dataframe)
training_targets = preprocess_targets(training_dataframe)

validation_examples = preprocess_features(validation_dataframe)
validation_targets = preprocess_targets(validation_dataframe)

_ = normalized_training_examples.hist(bins=20, figsize=(18, 12), xlabelsize=10)

print('Training examples summary:')
display.display(training_examples.describe())
print('Validation examples summary:')
display.display(validation_examples.describe())

print('Training targets summary:')
display.display(training_targets.describe())
print('Validation targets summary:')
display.display(validation_targets.describe())

normalized_dataframe = normalize_linear_scale(preprocess_features(california_housing_dataframe))
normalized_training_examples = normalized_dataframe.head(12000)
normalized_validation_examples = normalized_dataframe.tail(5000)

_, gradient_training_looses, gradient_validation_losses = train_nn_regression_model(
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
    steps=2000,
    batch_size=800,
    hidden_units=[10, 10, 8, 6, 4, 2],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

_, adagrad_training_losses, adagrad_validation_losses = train_nn_regression_model(
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.5),
    steps=2000,
    batch_size=50,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)

_, adam_training_losses, adam_validation_losses = train_nn_regression_model(
    my_optimizer=tf.train.AdamOptimizer(learning_rate=0.009),
    steps=2000,
    batch_size=50,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)

plt.ylabel("RMSE")
plt.xlabel("Periods")
plt.title("Root Mean Squared Error vs. Periods")
plt.plot(gradient_training_looses, label='Gradient training')
plt.plot(gradient_validation_losses, label='Gradient validation')
plt.plot(adagrad_training_losses, label='Adagrad training')
plt.plot(adagrad_validation_losses, label='Adagrad validation')
plt.plot(adam_training_losses, label='Adam training')
plt.plot(adam_validation_losses, label='Adam validation')
_ = plt.legend()
