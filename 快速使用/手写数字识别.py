import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

encode_input = tf.keras.Input(shape=(28,28,1), name='img')
h1 = layers.Conv2D(16, 3, activation='relu', padding='SAME')(encode_input)
h1 = layers.MaxPool2D()(h1)
h1 = layers.Conv2D(32, 3, activation='relu', padding='SAME')(h1)
h1 = layers.MaxPool2D()(h1)
h1 = layers.Flatten()(h1)
out = layers.Dense(10, activation='softmax')(h1)

model = tf.keras.Model(inputs=encode_input, outputs=out)

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=16)

print(model.predict(x_test, batch_size=8))
