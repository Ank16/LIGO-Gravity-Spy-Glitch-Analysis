#Import libraries
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers, activations
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Input, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import os

#Set some variables for data loading
train_dir = 'data/train'
val_dir = 'data/validation'
batch_size = 16
class_names = os.listdir(train_dir)

#Create generators that can load images in batches from the disk (prevents RAM from filling)
train_gen = ImageDataGenerator(rescale = 1./255.,
                           samplewise_center=True,
                           samplewise_std_normalization=True)

val_gen =  ImageDataGenerator(rescale = 1./255.,
                           samplewise_center=True,
                           samplewise_std_normalization=True)

train_gen = train_gen.flow_from_directory(train_dir,
                                  batch_size=batch_size,
                                  classes=class_names,
                                  class_mode='categorical',
                                  shuffle=True,
                                  target_size=(300,300),
                                  seed= 123)

val_gen = val_gen.flow_from_directory(val_dir,
                                  batch_size=batch_size,
                                  classes=class_names,
                                  class_mode='categorical',
                                  shuffle=False,
                                  target_size=(300,300),
                                  seed= 123)

#Define model structure
model = Sequential([
  layers.Flatten(input_shape=(300,300,3)),
  layers.Dense(32, activation='relu'),
  layers.Dense(64, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(22, activation='softmax')
])

#Compile model
model.compile(optimizer = Adam(0.001), 
              loss = 'categorical_crossentropy', metrics = ['accuracy', tf.keras.metrics.AUC(curve='PR')])
model.summary()

#Train model
history= model.fit(train_gen,
                 epochs=10,
                 validation_data=val_gen)
