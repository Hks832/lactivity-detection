#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import optimizers
from keras.layers import *
from keras.layers import GRU, RepeatVector
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# declare some constants
data_dir = "/Industrial Surveillance Dataset"
img_height, img_width = 64, 64  # dimension of each frame of videos
seq_len = 40  # number of images pass as one sequence
final_seq = int(seq_len / 5)
classes = ["Non-Violent", "Violent"]
roi_height, roi_width = 200, 200  # dimensions of ROI


# extraction of frames from videos
def frames_extraction(video_path, rois):
    frames_list = []
    vidObj = cv2.VideoCapture(video_path)
    count = 1

    while count <= seq_len:
        success, image = vidObj.read()
        if success:
            frames_roi = []
            for roi in rois:
                image_roi = image[roi[0]:roi[0] + roi_height, roi[1]:roi[1] + roi_width]
                image_roi = cv2.resize(image_roi, (img_height, img_width))
                image_roi = image_roi / 255
                frames_roi.append(image_roi)
            if count % 5 == 0:
                frames_list.append(np.concatenate(frames_roi, axis=2))
            count += 1
        else:
            break

    return frames_list, count


# data creation
def create_data(input_dir, rois):
    X = []
    Y = []

    classes_list = os.listdir(input_dir)

    for c in classes_list:
        print(c)
        if c in classes:
            if c == "Non-Violent":
                y = int(0)
            elif c == "Violent":
                y = int(1)
            else:
                print()
            files_list = os.listdir(os.path.join(input_dir, c))
            for f in files_list:
                frames, count = frames_extraction(os.path.join(os.path.join(input_dir, c), f), rois)

                if len(frames) == final_seq:
                    X.append(frames)
                    Y.append(y)

    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y


# Data augmentation
def augment_data(X):
    datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, horizontal_flip=True)
    augmented_data = []
    for x in X:
        x_aug = np.array([datagen.random_transform(img) for img in x])
        augmented_data.append(x_aug)
    return np.array(augmented_data)


# set the ROIs
rois = [(200, 200), (100, 100)]

# create the data
X, Y = create_data(data_dir, rois)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, shuffle=True, random_state=0)

# Augment training data
X_train_augmented = augment_data(X_train)
X_train = np.concatenate((X_train, X_train_augmented))
y_train = np.concatenate((y_train, y_train))

# Define the new input shape
input_shape = (8, 64, 64, len(rois) * 3)  # *3 because of 3 color channels per ROI

# C3D based model design
model = Sequential()

model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape, padding='same'))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid'))
model.add(BatchNormalization())
model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid'))
model.add(BatchNormalization())
model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same'))
model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid'))
model.add(BatchNormalization())
model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same'))
model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Simple CNN Model
model5=Sequential()
model5.add(Conv2D(32,(3,3),activation="relu",input_shape=(8,64*64,len(rois) * 3)))
model5.add(MaxPooling2D(2,2))
model5.add(Flatten())
model5.add(Dense(100,activation="relu"))
model5.add(Dense(10,activation="softmax"))

# RNN Model
model2 = Sequential()
model2.add(SimpleRNN(50, input_shape=(8, 64*64*len(rois) * 3), activation='relu'))
model2.add(Dense(1, activation='softmax'))

# LSTM Model [not enough memory to use this model...]
# model3 = Sequential()
# model3.add(ConvLSTM2D(filters = 64, kernel_size = (3, 3), return_sequences = True, data_format = "channels_last", input_shape = (final_seq, img_height, img_width, 3)))
# model3.add(ConvLSTM2D(filters = 128, kernel_size = (3, 3), return_sequences = True))
# model3.add(ConvLSTM2D(filters = 256, kernel_size = (3, 3), return_sequences = True))
# model3.add(TimeDistributed(Flatten()))
# model3.add(GRU(200))
# model3.add(Dense(256, activation="relu"))
# model3.add(Dropout(0.5))
# model3.add(Dense(128, activation="relu"))
# model3.add(Dropout(0.3))
# model3.add(Dense(64, activation="relu"))
# model3.add(Dropout(0.1))
# model3.add(Dense(1, activation='sigmoid'))
# model3.summary()

# LSTM 2
model4 = Sequential()
model4.add(LSTM(4, input_shape=(8, 64* 64* len(rois) * 3)))
model4.add(Dense(1))
model4.summary()

# 3D CNN Training
opt = Adam(0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[keras.metrics.BinaryAccuracy()])
earlystop = EarlyStopping(patience=50)
callbacks = [earlystop]
print("[INFO]...Model is training:")
history = model.fit(x=X_train, y=y_train, epochs=50, batch_size=8, shuffle=True, validation_split=0.10, callbacks=callbacks)

# Simple CNN Training
X_train_scnn = X_train.reshape((X_train.shape[0], X_train.shape[1],64*64, len(rois) * 3))
X_test_scnn = X_test.reshape((X_test.shape[0], X_test.shape[1],64*64, len(rois) * 3))
print("[SIMPLE CNN] training CNN...")
model5.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model5.fit(X_train_scnn,y_train,epochs=10)

# RNN Training
model2.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
print("[RNN] training RNN...")
X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], -1))
X_test_rnn = X_test.reshape((X_test.shape[0], X_test.shape[1], -1))
# model2.fit(X_train_rnn, y_train, epochs=20)

# LSTM Training
# opt = keras.optimizers.Adam(0.0001)
# model3.compile(loss='binary_crossentropy', optimizer=opt, metrics=[keras.metrics.BinaryAccuracy()])
# print("[LSTM] Model is Training...")
# earlystop = EarlyStopping(patience=50)
# callbacks = [earlystop]
# print("[INFO]...Model is training:")
# history = model3.fit(x = X_train, y = y_train, epochs=10, batch_size = 8 , shuffle=True, validation_split=0.10, callbacks=callbacks)


# LSTM 2 Trainingc
print("[LSTM 2] is training...")
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], -1))
model4.compile(loss='mean_squared_error', optimizer=opt, metrics=[keras.metrics.BinaryAccuracy()])
# model4.fit(X_train_lstm, y_train, epochs=50, batch_size=1, verbose=2, shuffle=True, validation_split=0.10, callbacks=callbacks)

# # make predictions on the test sets
c3_pred = model.predict(X_test)
# rnn_pred = model2.predict(X_test_rnn)
sim_snn = model5.predict(X_test_scnn)

# # Train the Meta learner
X_test_meta = np.column_stack((c3_pred, sim_snn))
# print("Meta Learning Output...", X_test_meta)

# # Train the meta-model on the combined feature matrix and the target values
meta_model = LinearRegression()
print("[Fitting] Meta Model")
meta_model.fit(X_test_meta, y_test)
x = meta_model.predict(X_test_meta)
print("ensemble", x, "v.s. normal", c3_pred)