import numpy as np
import tensorflow as tf


x = np.random.normal(size=[10,10])
y = np.random.normal(size=[10,10])
z = np.dot(x, y)

print(z)


x = tf.random_normal([10, 10])
y = tf.random_normal([10, 10])

z = tf.matmul(x, y)

sess  =tf.Session()
z_val = sess.run()

print(z_val)


