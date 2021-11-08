import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# 数据处理部分，包括训练集与测试集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 通过函数式构建网络模型
encode_input = tf.keras.Input(shape=(28,28,1), name='img')
# 第一层卷积，padding设置为SAME，保持分辨率
h1 = layers.Conv2D(16, 3, activation='relu', padding='SAME')(encode_input)
# 最大池化操作
h1 = layers.MaxPool2D()(h1)
# 第二层卷积
h1 = layers.Conv2D(32, 3, activation='relu', padding='SAME')(h1)
# 最大池化操作
h1 = layers.MaxPool2D()(h1)
# 将二维数据拉直成为一维数据
h1 = layers.Flatten()(h1)
# 第三层全连接操作
out = layers.Dense(10, activation='softmax')(h1)

model = tf.keras.Model(inputs=encode_input, outputs=out)

# 设置模型参数
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# 输入数据开始训练模型
model.fit(x_train, y_train, epochs=2, batch_size=16)

print("第一次输出：")
print(model.predict(x_test, batch_size=8))

# 保存完整模型
# .h5格式
# model.save("../weights/minist.h5")
#del model

# .pb模式
model.save("../weights/minist", save_format="tf")
del model
print("model saved!")

#model = tf.keras.models.load_model("../weights/minist.h5")
#print("第二次输出：")
#print(model.predict(x_test, batch_size=8))
