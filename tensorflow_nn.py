#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from xzc_tools import tools

try:
    # 定义数据batch的大小
    batch_size = 8

    # 定义神经网络的参数
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    # 存放训练数据的位置 x-特征  y-类别
    x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
    y_real = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

    # 定义神经网络的前向传播过程
    a = tf.matmul(x, w1)
    y_predict = tf.matmul(a, w2)

    # 定义损失函数和反向传播的算法
    y_predict = tf.sigmoid(y_predict)  # 求预测的类别

    cross_entropy = -tf.reduce_mean(y_predict * tf.log(tf.clip_by_value(y_predict, 1e-10, 1.0)) +
                                    (1 - y_predict) * tf.log(tf.clip_by_value(1 - y_predict, 1e-10, 1.0)))

    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # 通过随机数生成模拟数据集
    rdm = np.random.RandomState(1)
    dataset_size = 128
    X = rdm.rand(dataset_size, 2)
    Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

    # 创建会话
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        print(sess.run(w1))
        print(sess.run(w2))

        maxCycle = 10000
        for i in range(maxCycle):
            start = (i * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)

            # 选取样本数据
            sess.run(train_step, feed_dict={x: X[start:end], y_real: Y[start:end]})

            if i % 100 == 0:
                total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_real: Y})
                print(i, total_cross_entropy)
        print(sess.run(w1))
        print(sess.run(w2))
except Exception as msg:
    tools.printInfo(2, msg)