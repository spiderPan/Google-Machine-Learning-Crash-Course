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


def construct_feature_columns():
    return set([tf.feature_column.numeric_column('pixels', shape=784)])


def create_training_input_fn(features, labels, batch_size, num_epochs=None, shuffle=True):
    def _input_fn(num_epochs=None, shuffle=True):
        idx = np.random.permutation(features.index)
        raw_features = {'pixels': features.reindex(idx)}
        raw_targets = np.array(labels[idx])

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))
        ds = ds.batch(batch_size).repeat(num_epochs)

        if shuffle:
            ds = ds.shuffle(10000)

        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch
    return _input_fn


def create_perdict_input_fn(features, labels, batch_size):
    def _input_fn():
        raw_features = {'pixels': features.values}
        raw_targets = np.array(labels)

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))
        ds = ds.batch(batch_size)

        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch
    return _input_fn


def train_linear_regression_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    periods = 10
    steps_per_period = steps / periods

    predict_training_input_fn = create_perdict_input_fn(training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_perdict_input_fn(validation_examples, validation_targets, batch_size)
    training_input_fn = create_training_input_fn(training_examples, training_targets, batch_size)

    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    classifier = tf.estimator.LinearClassifier(feature_columns=construct_feature_columns(),
                                               n_classes=10,
                                               optimizer=my_optimizer,
                                               config=tf.estimator.RunConfig(keep_checkpoint_max=1))
    print('Training Model...')
    print('LogLoss error (on validation data):')
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        classifier.train(input_fn=training_input_fn, steps=steps_per_period)
        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)

        print(" period %02d : %0.2f" % (period, validation_log_loss))

        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print("Model trianing finished")
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents')))

    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])
    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)

    plt.ylabel('Logloss')
    plt.xlabel('Periods')
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label='Training')
    plt.plot(validation_errors, label='Validation')
    plt.legend()
    plt.show()

    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title('Confusion matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return classifier


def train_nn_regression_model(
        learning_rate,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    periods = 10
    steps_per_period = steps / periods

    predict_training_input_fn = create_perdict_input_fn(training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_perdict_input_fn(validation_examples, validation_targets, batch_size)
    training_input_fn = create_training_input_fn(training_examples, training_targets, batch_size)

    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    classifier = tf.estimator.DNNClassifier(feature_columns=construct_feature_columns(),
                                            hidden_units=hidden_units,
                                            n_classes=10,
                                            optimizer=my_optimizer,
                                            config=tf.estimator.RunConfig(keep_checkpoint_max=1))
    print('Training Model...')
    print('LogLoss error (on validation data):')
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        classifier.train(input_fn=training_input_fn, steps=steps_per_period)
        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)

        print(" period %02d : %0.2f" % (period, validation_log_loss))

        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print("Model trianing finished")
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents')))

    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])
    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)

    plt.ylabel('Logloss')
    plt.xlabel('Periods')
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label='Training')
    plt.plot(validation_errors, label='Validation')
    plt.legend()
    plt.show()

    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title('Confusion matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return classifier


mnist_dataframe = pd.read_csv('data/mnist_train_small.csv', sep=",", header=None)
mnist_test_dataframe = pd.read_csv('data/mnist_test.csv', sep=',', header=None)
mnist_dataframe = mnist_dataframe.head(10000)
mnist_dataframe = mnist_dataframe.reindex(np.random.permutation(mnist_dataframe.index))
display.display(mnist_dataframe.head())

training_targets, training_examples = parse_labels_and_features(mnist_dataframe[:7500])
testing_targets, testing_examples = parse_labels_and_features(mnist_test_dataframe)
display.display(training_examples.describe())

validation_targets, validation_examples = parse_labels_and_features(mnist_dataframe[7500:10000])
display.display(validation_examples.describe())

rand_example = np.random.choice(training_examples.index)
_, ax = plt.subplots()
ax.matshow(training_examples.loc[rand_example].values.reshape(28, 28))
ax.set_title('Label:%i' % training_targets.loc[rand_example])
ax.grid(False)

classifier = train_nn_regression_model(
    learning_rate=0.05,
    steps=1000,
    batch_size=30,
    hidden_units=[100, 100],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

test_predictions = classifier.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item['class_ids'][0] for item in test_predictions])

accuracy = metrics.accuracy_score(testing_targets, test_predictions)
print("Accuracy on test data: %0.2f" % accuracy)

print(classifier.get_variable_names())

weights0 = classifier.get_variable_value('dnn/hiddenlayer_0/kernel')

print('weighs 0 shape:', weights0.shape)
num_nodes = weights0.shape[1]
num_rows = int(math.ceil(num_nodes / 10.0))
fig, axes = plt.subplots(num_rows, 10, figsize=(20, 2 * num_rows))
for coef, ax in zip(weights0.T, axes.ravel()):
    # Weights in coef is reshaped from 1x784 to 28x28.
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.pink)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
