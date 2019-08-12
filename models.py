from __future__ import print_function

import librosa
import librosa.display

import matplotlib.pyplot as plt

from keras import models
from keras import layers
import numpy as np
import pandas as pd

import sklearn

# import freesound

from audioread import NoBackendError
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tqdm import tqdm
import glob, os


from pathlib import Path
import csv
import warnings  # record warnings from librosa
from sklearn.model_selection import train_test_split
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.activations import relu, sigmoid
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from sklearn.preprocessing import StandardScaler
class global_For_Class:
    input_shape= None




# model.add(layers.Dense(256, activation='relu', input_shape= global_For_Class.input_shape))
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform
import keras.backend as K
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

def create_model2(layers_l, batch_size):
    model = models.Sequential()
    model.add(
        layers.Dense(units=batch_size, activation='relu', input_shape=(X_train.shape[1],)))
    for i in range(layers_l):
        # if i == 0:
        #     # TODO: Add a dense layer
        #     # model.add(Activation(activation))
        # else:
        batch_size=2*batch_size
        model.add(layers.Dense(units=batch_size, activation='relu'))
        # TODO: Add a Dense later AND activation (see above)

    # TODO: Add last dense layer # Note: no activation beyond this point
    model.add(layers.Dense(1, activation='sigmoid'))  # the 1 means binary classification
    model.compile(optimizer='adam'
                  , loss='binary_crossentropy'
                  , metrics=['accuracy'])
    print("in iteration\n")
    print(layers_l, " ", " ", batch_size)
    return model


def create_model(X_train, Y_train, X_test, Y_test):
    model = models.Sequential()
    model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu', input_shape=(X_train.shape[1],)))
    choices={{choice(['one', 'two', 'three', 'four', 'five'])}}
    if choices == 'two':
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
    elif choices == 'three':
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
    elif choices == 'four':
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
    elif choices == 'five':
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
        model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # the 1 means binary classification
    model.compile(optimizer='adam'
                  , loss='binary_crossentropy'
                  , metrics=['accuracy'])
    model.fit(X_train, Y_train,
              batch_size={{choice([128, 256, 512])}},
              nb_epoch=20,
              verbose=2,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def best_result_search():
    """"
    important to use scaling only after splitting the data into train/validation/test
    scale on training set only, then use the returend "fit" parameters to scale validation and test
    """

    # keras
    #  how i expect to receive a model:
    # loaded model = jsonLoad \pandaLoad mori's choice on the saving format

    # model = models.Sequential()
    #
    # # print(X_train_kfold_scaled.shape[1])  #  45 (is the number of columns for each sample)
    # model.add(layers.Dense(200, activation='relu', input_shape=(X_train.shape[1],)))
    #
    # # model.add(layers.Dense(128, activation='relu'))
    #
    # model.add(layers.Dense(400, activation='relu'))
    #
    # model.add(layers.Dense(1, activation='sigmoid'))  # the 1 means binary classification
    #
    # model.compile(optimizer='adam'
    #               , loss='binary_crossentropy'
    #               , metrics=['accuracy'])
    #
    # # train model on training set of Kfold
    # history = model.fit(X_train,
    #                     y_train,
    #                     epochs=20,
    #                     batch_size=128)
    #
    # test_loss, test_acc = model.evaluate(X_test, y_test)
    #
    # print('test_acc in fold number : ', test_acc)
    #
    # results = model.evaluate(X_test, y_test)
    # print(f'results on the test data in fold number : ', results)

    model = KerasRegressor(build_fn=create_model, verbose=0)

    # layers = [[30], [20, 40], [15, 30, 40]]
    layers_sizes = [list(range(x)) for x in range(1, 2)]
    batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # activations = ['relu', 'softmax']
    param_grid = dict(layers_l=list(range(5)) , batch_size=batch_sizes, epochs=[20,30])
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error')

    grid_result = grid.fit(X_test, y_test)

    print([grid_result.best_score_, grid_result.best_params_])

    for scores in grid_result.cv_results_:
        print("%f (%f)" % (scores.mean(), scores.std()))

    print(grid_result.best_params_)


def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    path = Path('module_data')
    current_path_x_train = path / f"scream_train_x.pkl"
    with current_path_x_train.open('rb') as file:
        X_train = pkl.load(file)

    current_path_x_test = path / f"scream_test_x.pkl"
    with current_path_x_test.open('rb') as file:
        X_test = pkl.load(file)

    current_path_y_train = path / f"scream_train_y.pkl"
    with current_path_y_train.open('rb') as file:
        Y_train = pkl.load(file)

    current_path_y_test = path / f"scream_test_y.pkl"
    with current_path_y_test.open('rb') as file:
        Y_test = pkl.load(file)
    # x_train = X_train.reshape(60000, 784)
    # x_test = X_test.reshape(10000, 784)
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    # nb_classes = 1
    # y_train = np_utils.to_categorical(Y_train, nb_classes)
    # y_test = np_utils.to_categorical(Y_test, nb_classes)
    # return x_train, y_train, x_test, y_test
    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    # load from pickle test data
    # path = Path('module_data')
    # current_path_x_train = path / f"data_for_train_x.pkl"
    # with current_path_x_train.open('rb') as file:
    #     X_train = pkl.load(file)
    #
    # current_path_x_test = path / f"data_for_test_x.pkl"
    # with current_path_x_test.open('rb') as file:
    #     X_test = pkl.load(file)
    #
    # current_path_y_train = path / f"data_for_train_y.pkl"
    # with current_path_y_train.open('rb') as file:
    #     Y_train = pkl.load(file)
    #
    # current_path_y_test = path / f"data_for_test_y.pkl"
    # with current_path_y_test.open('rb') as file:
    #     Y_test = pkl.load(file)
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials(),
                                          eval_space=True)
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)



