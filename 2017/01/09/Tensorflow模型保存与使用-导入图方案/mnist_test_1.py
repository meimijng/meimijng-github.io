#!/usr/bin/env python
# 导入mnist数据库
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

import tensorflow as tf

# 创建会话
sess = tf.Session()

new_saver = tf.train.import_meta_graph('mnist_1/ckp.meta')
new_saver.restore(sess, 'mnist_1/ckp')

x = tf.get_collection('x')[0]
y = tf.get_collection('y')[0]

print("恢复模型成功！")

# 取出一个测试图片
idx = 0
img = mnist.test.images[idx]

# 根据模型计算结果
ret = sess.run(y, feed_dict = {x : img.reshape(1, 784)})

print("计算模型结果成功！")

# 显示测试结果
print("预测结果:%d"%(ret.argmax()))
print("实际结果:%d"%(mnist.test.labels[idx].argmax()))
