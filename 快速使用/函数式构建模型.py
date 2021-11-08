import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

input_x = tf.keras.Input(shape=(72,))
hidden1 = layers.Dense(32, activation="relu")(input_x)
print(hidden1.shape)
hidden2 = layers.Dense(32, activation="relu")(hidden1)
add_layer = layers.Add()([hidden1, hidden2])
pred = layers.Dense(10, activation="softmax")(add_layer)

model = tf.keras.Model(inputs=input_x, outputs=pred)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=['accuracy'])

train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))

val_x = np.random.random((200, 72))
val_y = np.random.random((200, 10))

model.fit(train_x, train_y, epochs=100, batch_size=100,
          validation_data=(val_x, val_y))

# 保存整个模型
model.save("all_model.h5")