"""
This is a program to compute and identify CIFAR 10 images. CIFAR or Canadian Institute For Advanced Research
provides 10 different categories of images in it. It has 60K of 32x32 images wherein 50k are for training and 
10k are for test. We plan on building an image classifier using Tensorflow's Keras api
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

#Here we import the CIFAR dataset
cifar10 = tf.keras.datasets.cifar10

#Dividing the data between Test and Train
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#to check if the data loaded successfully
#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#Using this we reduce the pixle value
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()


# setting the number of classes as k = 10 as CIFAR 10
K = len(set(y_train))
 
#print("number of classes:", K)

# input layer

#We create a Keras Tensor Instance giving it the shape or dimensions of the inputs
i = Input(shape=x_train[0].shape)

######################################################
#For My Reference: 
#A filter or a kernel in a conv2D layer has a height and a width. They are generally smaller than 
#the input image and so we move them across the whole image.

#A filter acts as a single template or pattern, which, when convolved across the input, finds similarities 
#between the stored template & different locations/regions in the input image.

#Batch Norm is a normalization technique done between the layers of a Neural 
#Network instead of in the raw data. It is done along mini-batches instead of the full data set. It serves to speed up training and use higher learning rates, making learning easier.

#Pooling layers are used to reduce the dimensions of the feature maps
#Max pooling is a maximazation function of the same
######################################################

#Conv2D is a 2 dimensional Convolution layer with parameters: 32 filters, Shape of the kernal (The shape of the window that scans smaller section of the image), Activation: ReLU Or Rectified Linear Unit func.
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
#Rough sample set created. Less defined Tensors are created here

#We keep increasing the defination to create more sample sets by increasing number of filters aka more passes
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
 
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
 
x = Flatten()(x)
x = Dropout(0.2)(x)
 
# Hidden layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
 
# last hidden layer i.e.. output layer
x = Dense(K, activation='softmax')(x)
 
model = Model(i, x)
 
# model description
model.summary()

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Fit
r = model.fit(
  x_train, y_train, validation_data=(x_test, y_test), epochs=50)


# Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='acc', color='red')
plt.plot(r.history['val_accuracy'], label='val_acc', color='green')
plt.legend()

# save the model
model.save('ChaiPaw.h5')
print("Model Successfully Saved!")