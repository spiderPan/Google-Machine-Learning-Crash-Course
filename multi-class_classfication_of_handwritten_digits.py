import glob
import io
import math
import os

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from IPython import display
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt
from sklearn import metrics
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


def parse_labels_and_features(dataset):
    labels = dataset[0]
    features = dataset.loc[:, 1:784]
    features = features / 255
    return labels, features


mnist_dataframe = pd.read_csv('data/mnist_train_small.csv', sep=",", header=None)
mnist_dataframe = mnist_dataframe.head(10000)
mnist_dataframe = mnist_dataframe.reindex(np.random.permutation(mnist_dataframe.index))
display.display(mnist_dataframe.head())

training_targets, training_examples = parse_labels_and_features(mnist_dataframe[:7500])
display.display(training_examples.describe())

validation_targets, validation_examples = parse_labels_and_features(mnist_dataframe[7500:10000])
display.display(validation_examples.describe())
