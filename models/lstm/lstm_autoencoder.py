#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
LSTM auto encoder model which is created by using keras
'''

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed


def get_lstm_autoencoder_model(timesteps,
                               features,
                               encoding_dimension,
                               activation,
                               loss,
                               optimizer):
    """
    Create linear stack of layers using keras in order to create LSTM auto encoder model
    :param timesteps: window size
    :param features: data frame's columns
    :param encoding_dimension: encoding_dimension
    :param activation: user's choice for activation function
    :param loss: user's choice for loss function
    :param optimizer: user's choice for an optimizer
    :return: LSTM auto encoder model
    """

    model = Sequential()
    model.add(LSTM(encoding_dimension, activation=activation, input_shape=(timesteps, features)))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(encoding_dimension, activation=activation, return_sequences=True))
    model.add(TimeDistributed(Dense(features)))
    model.compile(optimizer=optimizer, loss=loss)

    return model
