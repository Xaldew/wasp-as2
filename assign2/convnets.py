#!/usr/bin/env python

import sys
import os
import locale
import random
import argparse
import warnings
import scipy
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score


def chollet_model(n_unique_words, n_dim, max_review_length):
    """Compile the Chollet model.

    """
    model = Sequential()
    model.add(layers.Embedding(n_unique_words, n_dim, input_length=max_review_length))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def krohn_model(n_unique_words, n_dim, max_review_length, dropout, embed_dropout):
    """Compile the Krohn model.

    """
    # convolutional layer architecture:
    n_conv = 256 # filters, a.k.a. kernels
    k_conv = 3 # kernel length

    # dense layer architecture:
    n_dense = 256

    model = Sequential()
    model.add(layers.Embedding(n_unique_words, n_dim, input_length=max_review_length)) 
    model.add(layers.SpatialDropout1D(embed_dropout))
    model.add(layers.Conv1D(n_conv, k_conv, activation='relu'))
    # model.add(layers.Conv1D(n_conv, k_conv, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(n_dense, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def get_model(enum, n_unique_words, max_review_length, dropout):
    """Retrieve a model based on the given enum.

    .. Keyword Arguments:
    :param enum: A string identifying a model to use.

    .. Returns:
    :returns: A compiled keras model.

    """
    if enum == "chollet":
        return chollet_model(n_unique_words, 128, max_review_length)
    elif enum == "krohn":
        return krohn_model(n_unique_words, 64, max_review_length, dropout, dropout)
    else:
        return chollet_model(n_unique_words, 128, max_review_length)


def main(output_dir, enum,
         batch_size, epochs, train_size,
         n_unique_words, max_review_length, dropout):
    """Perform Deep Learning on the IMDB test dataset.

    """
    print('Loading data...')
    (x_train, y_train), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words)
    print(len(x_train), 'training sequences')
    print(len(x_valid), 'validation sequences')

    pad_type = trunc_type = 'pre'
    idx = random.sample(range(0, len(x_train)), train_size)
    idx.sort()

    x_train = [x_train[i] for i in idx]
    y_train = [y_train[i] for i in idx]
    x_train = pad_sequences(x_train, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)
    x_valid = pad_sequences(x_valid, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)
    print('x_train shape:', x_train.shape)
    print('x_valid shape:', x_valid.shape)

    # Compile and train the model.
    model = get_model(enum, n_unique_words, max_review_length, dropout)
    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    output_fmt = os.path.join(output_dir, "weights.{epoch:02d}.hdf5")
    model_cps = ModelCheckpoint(filepath=output_fmt)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    history = model.fit(x_train, y_train,
                        verbose=1,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_valid, y_valid),
                        callbacks=[model_cps])

    # Plot accuracy and loss.
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    return 0


def parse_arguments(argv):
    """Parse the given argument vector.

    .. Keyword Arguments:
    :param argv: The arguments to be parsed.

    .. Types:
    :type argv: A list of strings.

    .. Returns:
    :returns: The parsed arguments.
    :rtype: A argparse namespace object.

    """
    fmtr = argparse.RawDescriptionHelpFormatter
    kdesc = "Convolutional Neural Network Testing"
    parser = argparse.ArgumentParser(description=kdesc, formatter_class=fmtr)
    parser.add_argument("output_dir", metavar="FILE", type=str,
                        default="model_output/conv",
                        help="Model output directory.")
    parser.add_argument("-model", "--model", metavar="M", type=str,
                        default="chollet", choices={"chollet", "krohn"},
                        help="Which Neural Network model to use.")
    parser.add_argument("-t", "--training-size", metavar="N", type=int,
                        default=10000,
                        help="Amount of training data to keep.")
    parser.add_argument("-d", "--dropout", metavar="V", type=float,
                        default=0.2,
                        help="Amount of dropout to use.")
    parser.add_argument("-n", "--unique-words", metavar="N", type=int,
                        default=10000,
                        help="Maximum number of unique words in the dataset.")
    parser.add_argument("-e", "--epochs", metavar="N", type=int,
                        default=10,
                        help="Number of epochs to train the model for.")
    parser.add_argument("-b", "--batch-size", metavar="N", type=int,
                        default=128,
                        help="Size of each mini-batch for the learning.")
    parser.add_argument("-m", "--max-review-length", metavar="N", type=int,
                        default=500,
                        help="Size of each mini-batch for the learning.")
    parser.add_argument("-s", "--seed", action="store", type=int, default=2,
                        help="The seed to use for splitting the training data.")
    parser.add_argument("-f", "--final", action="store", type=int, default=1,
                        help="Final epoch to use for evaluating the model.")

    return parser.parse_args(argv)


if __name__ == "__main__":
    ARGS = parse_arguments(sys.argv[1:])
    locale.setlocale(locale.LC_ALL, "")
    random.seed(ARGS.seed)
    sys.exit(main(ARGS.output_dir,
                  ARGS.model,
                  ARGS.batch_size,
                  ARGS.epochs,
                  ARGS.training_size,
                  ARGS.unique_words,
                  ARGS.max_review_length,
                  ARGS.dropout))
