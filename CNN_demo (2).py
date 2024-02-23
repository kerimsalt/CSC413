#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import mplfinance as mpf
#在cmd中pip install keras
from tensorflow.keras import backend as K
from collections import Counter


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import numpy as np
import os
import pandas as pd




savefig_dir = 'C:/Users/kerim/PycharmProjects/CSC413' # 64*64 graph npz path

# xx = input('enter the forecast horizon: \n')
os.chdir(savefig_dir)

# data = np.load('data' + xx + '.npz')
data=np.load('data3.npz')#name of graph npz


X_train=data['X_train']
Y_train=data['Y_train']
X_valid=data['X_valid']
Y_valid=data['Y_valid']
X_test= data['X_test']
Y_test= data['Y_test']


X_train=X_train / 255.0
X_test=X_test / 255.0
X_valid=X_valid / 255.0





model = models.Sequential()

model.add(layers.Conv2D(64,(3,3),strides=(1,1),input_shape=(64,64,4),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(layers.Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(layers.Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(layers.Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(layers.Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(4096,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1000,activation='softmax'))




# model = models.Sequential()
#
# model.add(layers.Conv2D(64,(3,3),strides=(1,1),input_shape=(128,128,3),padding='same',activation='relu',kernel_initializer='uniform'))
# model.add(layers.Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
# model.add(layers.MaxPooling2D(pool_size=(2,2)))
#
# model.add(layers.Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
# model.add(layers.Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
# model.add(layers.MaxPooling2D(pool_size=(2,2)))

# model.add(layers.Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
# model.add(layers.Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
# model.add(layers.Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
# model.add(layers.MaxPooling2D(pool_size=(2,2)))
#
# model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
# model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
# model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
# model.add(layers.MaxPooling2D(pool_size=(2,2)))

# model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
# model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
# model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
# model.add(layers.MaxPooling2D(pool_size=(2,2)))
#
# model.add(layers.Flatten())
# model.add(layers.Dense(4096,activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(4096,activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(1000,activation='softmax'))

from tensorflow.keras.models import model_from_json
from keras import optimizers
from keras.models import Sequential
optm = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optm,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              #loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               #loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])


history = model.fit(X_train, Y_train, epochs=100, validation_data=(X_valid, Y_valid),shuffle=True)
# input('hold on')
# ####
# import torch
# # save model
# torch.save(model.state_dict(), savefig_dir)
# ####
# input('check 1')
# # load model
# model = ModelNet(*args, **kwargs)
# model.load_state_dict(torch.load(savefig_dir))
# model.eval()

# # Calling `save('my_model')` creates a SavedModel folder `my_model`.
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
tmp = "model.h_"
model.save_weights(tmp)
# model.save_weights("model.h3")
print("Saved model to disk")
# # It can be used to reconstruct the model identically.
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(tmp)
# loaded_model.load_weights("model.h3")
print("Loaded model from disk")
# model.save("Bist_CNN.h5")
# loaded_model = load_model("network.h5")

test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)

pre=model.predict(X_test)
pre_result=list(map(lambda x:np.argmax(x),pre))


print("test_acc", test_acc)



from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

y_true = list(Y_test)
y_pred = pre_result
labels=[0,1] # labels = [0,1,2]

C2 = confusion_matrix(y_true, y_pred, labels=labels)


plt.matshow(C2, cmap=plt.cm.Reds)


for i in range(len(C2)):
    for j in range(len(C2)):
        plt.annotate(C2[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')


plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# input('hold on ')

plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
