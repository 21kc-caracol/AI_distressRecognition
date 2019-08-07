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

class global_For_Class:
    input_shape= None

def modelCreate :
    model = models.Sequential()

    # print(X_train_kfold_scaled.shape[1])  #  45 (is the number of columns for each sample)
    model.add(layers.Dense(256, activation='relu', global_For_Class.input_shape,)))

    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))  # the 1 means binary classification

    model.compile(optimizer='adam'
                  , loss='binary_crossentropy'
                  , metrics=['accuracy'])

    # train model on training set of Kfold
    history = model.fit(X_train_kfold_scaled,
                        y_train_kfold,
                        epochs=20,
                        batch_size=128)
    test_loss, test_acc = model.evaluate(X_test_kfold_scaled, y_test_kfold)

    print(f'test_acc in fold number {fold}: ', test_acc)

    results = model.evaluate(X_test_scaled, y_test_loaded)
    print(f'results on the test data in fold number {fold}: ', results);


model.add(layers.Dense(256, activation='relu', input_shape= global_For_Class.input_shape))





