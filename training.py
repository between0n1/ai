import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

train_file = 'recycled_32_train.npz'
test_file = 'recycled_32_test.npz'

with np.load(train_file) as data:
    image = data['x']
    label = data['y']

with np.load(test_file) as data:
    test_image = data['x']
    test_label = data['y']


train_dataset = tf.data.Dataset.from_tensor_slices((image, label))
test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_label))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(3,32, 32)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_dataset, epochs=10)

print(model.evaluate(test_dataset))