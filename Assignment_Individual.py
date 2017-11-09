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
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam


#%% read MNIST kaggle data

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

train_x, valid_x, train_y, valid_y = train_test_split(data_train.ix[:,1:], data_train.ix[:,0], test_size=0.20, random_state=13)

train_n = train_x.shape[0]
valid_n = valid_x.shape[0]
test_n  = data_test.shape[0]

train_x = (train_x.values).astype('float32') / 255.0
valid_x = (valid_x.values).astype('float32') / 255.0
test_x = (data_test.values).astype('float32') / 255.0
          
train_y = train_y.values.astype('int32')
valid_y = valid_y.values.astype('int32')

#%% plot data

train_x_img = train_x.reshape(train_n,  28, 28)

plt.imshow(train_x_img[9], cmap=plt.get_cmap('gray'))
plt.show()

#%% reshape variables to 28x28
train_x = train_x.reshape(train_n, 28, 28, 1)
valid_x = valid_x.reshape(valid_n, 28, 28, 1)
test_x = test_x.reshape(test_n, 28, 28, 1)

train_y = np_utils.to_categorical(train_y)
valid_y = np_utils.to_categorical(valid_y)


#%% Build MLP model

def CNN_0():
    model = Sequential()
    
    #create fully connected layers
    model.add(Flatten(input_shape = (28,28,1)))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    #softmax classifier
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-3), metrics = ["accuracy"])
    
    return model


#%%
model0 = CNN_0()
model0_hist = model0.fit(train_x, train_y, batch_size = 32, epochs=20,
                         validation_data=(valid_x, valid_y))

m0 = model0_hist.history

#%% MLP with dropout

def CNN_0_d():
    model = Sequential()
    
    #create fully connected layers
    model.add(Flatten(input_shape = (28,28,1)))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    #softmax classifier
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-3), metrics = ["accuracy"])
    
    return model

model0_d = CNN_0_d()
model0_d_hist = model0_d.fit(train_x, train_y, batch_size = 32, epochs=20,
                         validation_data=(valid_x, valid_y))

m0_d = model0_hist.history

#%% run model on test data

pred0 = model0.predict_classes(test_x, batch_size=64)

test0_y = pd.DataFrame({"ImageId": list(range(1,len(pred0)+1)),
                         "Label": pred0})
test0_y.to_csv("test_pred0.csv", index=False, header=True)

#%% Plot the difference
plt.style.use('ggplot')
plt.figure(figsize=(10,5))
plt.plot(m0['val_acc'][:20])
plt.plot(m0_d['val_acc'])
plt.legend(['Without Dropout', 'With Dropout'])
plt.xticks(range(0,20))
plt.ylim((0.94,1))
plt.title('MLP validation accuracy')
plt.show()

#%% CNN 1: Add more convolutional layer and dropout to avoid overfitting
def CNN_1():
    model = Sequential()
    
    #first convolution
    model.add(Conv2D(6, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    #model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2,2)))
    
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics = ["accuracy"])
    
    return model

#%% Train model 1

model1 = CNN_1()
model1_hist = model1.fit(train_x, train_y, batch_size = 64, epochs=20,
                         validation_data=(valid_x, valid_y))

m1 = model1_hist.history

#%% run model on test data

pred1 = model1.predict(test_x, batch_size=64)

test1_y = pd.DataFrame({"ImageId": list(range(1,len(pred1)+1)),
                         "Label": pred1})
test1_y.to_csv("test_pred1.csv", index=False, header=True)


#%% Model 2: Tensorflow tutorial

def CNN_2():
    model = Sequential()
    
    #first convolution
    model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     padding = 'same',
                     input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2,2)))
    
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), 
                  metrics = ["accuracy"])
    
    return model

#%% Train model 2

model2 = CNN_2()
model2_hist = model2.fit(train_x, train_y, batch_size = 50, epochs=20,
                         validation_data=(valid_x, valid_y))

m2 = model2_hist.history

#%% run model 2 on test data

pred2 = model2.predict(test_x, batch_size=64)

test2_y = pd.DataFrame({"ImageId": list(range(1,len(pred2)+1)),
                         "Label": pred2})
test2_y.to_csv("test_pred2.csv", index=False, header=True)




#%% Model 2: change the kernel size to 3

def CNN_3():
    model = Sequential()
    
    #first convolution
    model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     padding='same',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(1,1)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), 
                  metrics = ["accuracy"])
    
    return model

#%% Train model 2

model3 = CNN_3()
model3_hist = model3.fit(train_x, train_y, batch_size = 50, epochs=20,
                         validation_data=(valid_x, valid_y))

m3 = model3_hist.history

#%% Plot the model 1, 2 and 3 accuracies

plt.style.use('ggplot')
plt.figure(figsize=(15,7))
plt.plot(m1['val_acc'])
plt.plot(m2['val_acc'])
plt.plot(m3['val_acc'])
plt.legend(["CNN-1", "CNN-2", "CNN-3"], loc=4)
plt.xticks(range(0,20))
plt.ylim((0.94,1))
plt.title('CNN models validation accuracy')
plt.show()


#%% Data augmentation

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=20,
                             height_shift_range=0.1,
                             width_shift_range=0.1)
# fit parameters from data
datagen.fit(train_x)

model2_hist_aug = model2.fit_generator(datagen.flow(train_x, train_y,
                                  batch_size=50 ),
                        steps_per_epoch=train_x.shape[0] // 50,
                        epochs=100,
                        validation_data=(valid_x, valid_y))

m2_aug = model2_hist_aug.history

#%%
pred2 = model2.predict_classes(test_x, batch_size=50)

test2_y = pd.DataFrame({"ImageId": list(range(1,len(pred2)+1)),
                         "Label": pred2})
test2_y.to_csv("test_pred2_aug2.csv", index=False, header=True)


#%% Data augmentation with model 3

model3_hist_aug = model3.fit_generator(datagen.flow(train_x, train_y,
                                  batch_size=50 ),
                        steps_per_epoch=train_x.shape[0] // 50,
                        epochs=100,
                        validation_data=(valid_x, valid_y))

m3_aug = model3_hist_aug.history

#%% test model 3
pred3_aug = model3.predict_classes(test_x, batch_size=50)

test3_y_aug = pd.DataFrame({"ImageId": list(range(1,len(pred3_aug)+1)),
                         "Label": pred3_aug})
test3_y_aug.to_csv("test_pred3_aug3.csv", index=False, header=True)


#%% Plot the model 2 and 3 accuracies

#with sns.axes_style("white"):
plt.style.use('ggplot')
plt.figure(figsize=(15,7))
plt.plot(m2['val_acc'], linestyle="--")
plt.plot(m3['val_acc'], linestyle="--")
plt.plot(m2_aug['val_acc'][:20])
plt.plot(m3_aug['val_acc'][:20])
plt.legend(["CNN-2", "CNN-3", "CNN-2 augmented", "CNN-3 augmented"], loc=4)
plt.xticks(range(0,20))
plt.ylim((0.97,1))
plt.title('CNN models validation accuracy')
plt.show()


