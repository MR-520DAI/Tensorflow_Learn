import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name = "my_model")
        self.num_classes = num_classes
        self.layer1 = layers.Dense(32, activation="relu")
        self.layer2 = layers.Dense(num_classes, activation="softmax")
    def call(self, inputs):
        h1 = self.layer1(inputs)
        out = self.layer2(h1)
        return out
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

model = MyModel(num_classes=10)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))
test_x = np.random.random((10, 72))

model.fit(train_x, train_y, batch_size=64, epochs=50)

print(model.predict(test_x))

model.save("../weights/shanchu", save_format="tf")
