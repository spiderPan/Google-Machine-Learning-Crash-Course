import collections
import math

import matplotlib.pyplot as pyt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)
train_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/train.tfrecord'
train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
test_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/test.tfrecord'
test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)


def _parse_function(record):
    features = {
        "terms": tf.VarLenFeature(dtype=tf.string),
        "labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32)
    }
    parsed_features = tf.parse_single_example(record, features)
    terms = parsed_features['terms'].values
    labels = parsed_features['labels']

    return {"terms": terms}, labels


def _input_fn(input_filename, num_epoches=None, shuffle=True):
    ds = tf.data.TFRecordDataset(input_filename)
    ds = ds.map(_parse_function)

    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.padded_batch(25, ds.output_shapes)
    ds = ds.repeat(num_epoches)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


ds = tf.data.TFRecordDataset(train_path)
ds = ds.map(_parse_function)

n = ds.make_one_shot_iterator().get_next()
sess = tf.Session()
sess.run(n)

informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
                     "excellent", "poor", "boring", "awful", "terrible",
                     "definitely", "perfect", "liked", "worse", "waste",
                     "entertaining", "loved", "unfortunately", "amazing",
                     "enjoyed", "favorite", "horrible", "brilliant", "highly",
                     "simple", "annoying", "today", "hilarious", "enjoyable",
                     "dull", "fantastic", "poorly", "fails", "disappointing",
                     "disappointment", "not", "him", "her", "good", "time",
                     "?", ".", "!", "movie", "film", "action", "comedy",
                     "drama", "family")
terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="terms", vocabulary_list=informative_terms)

my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

terms_embedding_column = tf.feature_column.embedding_column(categorical_column=terms_feature_column, dimension=2)
feature_columns = [terms_embedding_column]
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[20, 20],
                                        optimizer=my_optimizer,
                                        model_dir="./dnn_model")

try:
    classifier.train(input_fn=lambda: _input_fn([train_path]), steps=1000)
    print(classifier.get_variable_names())
    print(classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights').shape)
    evaluation_metrics = classifier.evaluate(input_fn=lambda: _input_fn([train_path]),
                                             steps=1000)

    print("Training set metrics:")
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
    print('---')

    evaluation_metrics = classifier.evaluate(input_fn=lambda: _input_fn([test_path]), steps=1000)

    print('Test set metrics:')
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
    print('---')
except ValueError as err:
    print(err)
