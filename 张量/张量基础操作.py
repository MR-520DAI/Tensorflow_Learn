import tensorflow as tf

# 定义常量
t1 = tf.constant([1, 2, 3], dtype=tf.float32)
print(t1)

# 定义随机矩阵
t2 = tf.random.uniform(shape=(3,))
print(t2)

# 加法操作
print(tf.add(t1, t2))

# 定义矩阵
M1 = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
M2 = tf.constant([[2, 3], [4, 5]], dtype=tf.float32)

# 对应元素相乘
print(tf.multiply(M1, M2))
# 矩阵乘法
print(tf.matmul(M1, M2))