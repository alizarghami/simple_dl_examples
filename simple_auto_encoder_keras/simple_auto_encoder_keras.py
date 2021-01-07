#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 22:19:50 2020

@author: ali
"""

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam

import matplotlib.pyplot as plt


BATCH_SIZE = 1024
EPOCHS = 20


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

Z_train = X_train.reshape(-1, 784)
Z_test = X_test.reshape(-1, 784)

simple_auto_encoder = Sequential()

simple_auto_encoder.add(Dense(512, activation='elu', input_shape=(784,)))
simple_auto_encoder.add(Dense(128, activation='elu'))
simple_auto_encoder.add(Dense(10, activation='linear', name='bottleneck'))
simple_auto_encoder.add(Dense(128, activation='elu'))
simple_auto_encoder.add(Dense(512, activation='elu'))
simple_auto_encoder.add(Dense(784))

simple_auto_encoder.compile(Adam(), loss='mean_squared_error')

image = Z_test[0].reshape(28, 28)
res = simple_auto_encoder.predict(Z_test[0].reshape(-1, 784))
res = res.reshape(28, 28)

fig1 = plt.figure('Before training')
ax1 = fig1.add_subplot(1,2,1)
ax1.imshow(image)
ax2 = fig1.add_subplot(1,2,2)
ax2.imshow(res)


trained_model = simple_auto_encoder.fit(Z_train, Z_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(Z_test, Z_test))


res = simple_auto_encoder.predict(Z_test[0].reshape(-1, 784))
res = res.reshape(28, 28)

fig2 = plt.figure('After training')
ax1 = fig2.add_subplot(1,2,1)
ax1.imshow(image)
ax2 = fig2.add_subplot(1,2,2)
ax2.imshow(res)


simple_auto_encoder.save('models/model.h5')