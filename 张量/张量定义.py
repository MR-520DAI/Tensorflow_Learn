import tensorflow as tf

# 定义常量
t1 = tf.constant([1, 2, 3], dtype=tf.float32)
print(t1)

# 定义随机矩阵
t2 = tf.random.uniform(shape=(3,))
print(t2)

