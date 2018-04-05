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


california_housing_dataframe = pd.read_csv('data/california_housing_train.csv', sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))

training_dataframe = california_housing_dataframe.head(12000)
training_examples = preprocess_features(training_dataframe)
training_targets = preprocess_targets(training_dataframe)

validation_dataframe = california_housing_dataframe.tail(50000)
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
