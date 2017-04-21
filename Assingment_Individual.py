# -*- coding: utf-8 -*-
"""
Advanced Machine Learning - Individual Project
MNIST data set

Created on Thu Apr 20 17:36:38 2017

@author: Dinan Amiendiartha
"""
#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam, SGD


#%%

data_train = pd.read_csv('train.csv')
data_test = (pd.read_csv('test.csv').values).astype('float32')

train_x, valid_x, train_y, valid_y = train_test_split(data_train.ix[:,1:], data_train.ix[:,0], test_size=0.30, random_state=13)

train_n = train_x.shape[0]
valid_n = valid_x.shape[0]

train_x = (train_x.values).astype('float32') / 255.0
valid_x = (valid_x.values).astype('float32') / 255.0

train_y = train_y.values.astype('int32')
valid_y = valid_y.values.astype('int32')

#%%
train_x_img = train_x.reshape(train_n,  28, 28)

plt.imshow(train_x_img[9], cmap=plt.get_cmap('gray'))
plt.show()

#%% Building MLP model
params = {
            'solver': ['sgd', 'lbfgs', 'adam'],
            'learning_rate_init': [.1,.01,.001]
          }

gs_mlp = GridSearchCV(MLPClassifier(hidden_layer_sizes=(50,), max_iter=10), param_grid = params, cv = 5)
gs_mlp.fit(train_x, train_y)

gs_mlp.best_params_

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='adam', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.001)


#%% test on validation set

gs_mlp.score(train_x, train_y) # accuracy: 91.1%
gs_mlp.score(valid_x, valid_y) # accuracy: 88.5%

#%% Build CNN model

def CNN_0():
    model = Sequential()
    
    #first convolution
    model.add(Conv2D(32, kernel_size = (5, 5), input_shape = (28, 28, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    #second convolution
    model.add(Conv2D(64, kernel_size = (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), (2,2)))
    
    #create fully connected layers
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    
    #softmax classifier
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-3), metrics = ["accuracy"])
    
    return model

#%% train CNN model
train_x = train_x.reshape(train_n, 28, 28, 1)
valid_x = valid_x.reshape(valid_n, 28, 28, 1)

train_y = np_utils.to_categorical(train_y)
valid_y = np_utils.to_categorical(valid_y)

model0 = CNN_0()
model0.fit(train_x, train_y, batch_size = 64, epochs=5)

model0.get_weights