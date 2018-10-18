#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 ishitaku5522<ishitaku5522@gmail.com>
#
# Distributed under terms of the MIT license.

# Reference: https://arxiv.org/abs/1703.03400

import os
#  import sys
import numpy as np
import tensorflow as tf
#  import pandas as pd
import matplotlib.pyplot as plt

_script_dir = os.path.dirname(os.path.realpath(__file__))


def func(x, phase, amp):
    return np.sin(x + phase) * amp
    #  return x * np.sin(x) + x / 2


class NeuralNet(object):
    def __init__(self):
        self.W = {}
        self.b = {}

    def forward_func(self, x, weights, reuse=False, name="model"):
        with tf.name_scope(name):
            layer = x

            key = "0"
            layer = tf.nn.leaky_relu(
                tf.matmul(layer, weights["W_" + key]) + weights["b_" + key])

            key = "1"
            layer = tf.nn.leaky_relu(
                tf.matmul(layer, weights["W_" + key]) + weights["b_" + key])

            key = "out"
            layer = tf.matmul(layer, weights["W_" + key]) + weights["b_" + key]
        return layer

    def loss_func(self, pred, actual):
        ans = tf.reduce_mean(tf.square(pred - actual))
        return ans


def generate_test_dataset(one_set_size, x_sample_size, x_range, y_range):
    x_pre = []
    y_pre = []
    x_meta = []
    y_meta = []

    tmp_x = np.random.choice(
        np.arange(-x_range, x_range, 0.01),
        one_set_size * x_sample_size,
        replace=False)
    tmp_pre_x = np.random.choice(tmp_x, x_sample_size, replace=False)
    tmp_meta_x = np.setdiff1d(tmp_x, tmp_pre_x)
    phase = (np.random.sample() * 2 - 1) * np.pi
    amp = np.random.sample() * y_range
    tmp_pre_y = func(tmp_pre_x, phase, amp)
    tmp_meta_y = func(tmp_meta_x, phase, amp)
    x_pre.append(tmp_pre_x)
    y_pre.append(tmp_pre_y)
    x_meta.append(tmp_meta_x)
    y_meta.append(tmp_meta_y)

    x_pre = np.array(x_pre) * 0.8 / x_range
    y_pre = np.array(y_pre) * 0.8 / y_range
    x_meta = np.array(x_meta) * 0.8 / x_range
    y_meta = np.array(y_meta) * 0.8 / y_range

    x_pre = x_pre[:, :, np.newaxis]
    y_pre = y_pre[:, :, np.newaxis]
    x_meta = x_meta[:, :, np.newaxis]
    y_meta = y_meta[:, :, np.newaxis]

    return x_pre, x_meta, y_pre, y_meta, amp, phase


def generate_dataset(num_tasks, one_set_size, x_sample_size, x_range,
                     y_range):
    x_pre = []
    y_pre = []
    x_meta = []
    y_meta = []
    for i in range(num_tasks):
        tmp_x = np.random.choice(
            np.arange(-x_range, x_range, 0.01),
            one_set_size * x_sample_size,
            replace=False)
        tmp_pre_x = np.random.choice(tmp_x, x_sample_size, replace=False)
        tmp_meta_x = np.setdiff1d(tmp_x, tmp_pre_x)
        phase = (np.random.sample() * 2 - 1) * np.pi
        amp = np.random.sample() * y_range
        tmp_pre_y = func(tmp_pre_x, phase, amp)
        tmp_meta_y = func(tmp_meta_x, phase, amp)
        x_pre.append(tmp_pre_x)
        y_pre.append(tmp_pre_y)
        x_meta.append(tmp_meta_x)
        y_meta.append(tmp_meta_y)

    x_pre = np.array(x_pre) * 0.8 / x_range
    y_pre = np.array(y_pre) * 0.8 / y_range
    x_meta = np.array(x_meta) * 0.8 / x_range
    y_meta = np.array(y_meta) * 0.8 / y_range

    x_pre = x_pre[:, :, np.newaxis]
    y_pre = y_pre[:, :, np.newaxis]
    x_meta = x_meta[:, :, np.newaxis]
    y_meta = y_meta[:, :, np.newaxis]

    return x_pre, x_meta, y_pre, y_meta


def main():

    num_preupdates = 10
    one_set_size = 10
    num_tasks = 200
    x_sample_size = 30
    x_range = 10
    y_range = 10

    modelname = "sindataset"
    restore_epoch = 2300
    train_epoch = 0000
    task_batch_size = 1

    x_pre, x_meta, y_pre, y_meta = generate_dataset(
        num_tasks, one_set_size, x_sample_size, x_range, y_range)

    x_all = np.concatenate([x_pre, x_meta], axis=1)
    y_all = np.concatenate([y_pre, y_meta], axis=1)

    print(np.max(x_all), np.min(x_all))
    print(np.max(y_all), np.min(y_all))
    print(x_pre.shape, y_pre.shape, x_meta.shape, y_meta.shape)
    print(x_pre.shape, y_pre.shape, x_meta.shape, y_meta.shape)

    # [task, batch_for_a_task, input]
    x_pre_ph = tf.placeholder(tf.float32, shape=[None, None, 1])
    y_pre_ph = tf.placeholder(tf.float32, shape=[None, None, 1])
    x_meta_ph = tf.placeholder(tf.float32, shape=[None, None, 1])
    y_meta_ph = tf.placeholder(tf.float32, shape=[None, None, 1])

    dataset = tf.data.Dataset.from_tensor_slices((x_pre_ph, y_pre_ph, x_meta_ph, y_meta_ph))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=10)
    dataset = dataset.batch(task_batch_size)
    iterator = dataset.make_initializable_iterator()
    elements = iterator.get_next()
    X_pre, Y_pre, X_meta, Y_meta = elements

    model = NeuralNet()

    with tf.variable_scope("weights"):
        weights = {}

        key = "0"
        weights["W_" + key] = tf.get_variable(
            "W_" + key, shape=[1, 20])
        weights["b_" + key] = tf.get_variable("b_" + key, shape=[20])

        key = "1"
        weights["W_" + key] = tf.get_variable("W_" + key, shape=[20, 20])
        weights["b_" + key] = tf.get_variable("b_" + key, shape=[20])

        key = "out"
        weights["W_" + key] = tf.get_variable("W_" + key, shape=[20, 1])
        weights["b_" + key] = tf.get_variable("b_" + key, shape=[1])

    fewshot_lr = tf.constant(0.01)

    # Define few shot graphs
    fewshot_pred_op = model.forward_func(X_pre[0], weights, name="pre0")
    fewshot_loss_op = model.loss_func(fewshot_pred_op, Y_pre[0])

    pre_firstshot_loss_op = fewshot_loss_op

    with tf.name_scope("pre_grads0"):
        fewshot_grads = {}
        for key in weights:
            fewshot_grads[key] = tf.gradients(fewshot_loss_op, weights[key])
            fewshot_grads[key] = fewshot_grads[key][0]

    fewshot_weights = {}
    with tf.name_scope("pre_weights0"):
        for key in weights:
            fewshot_weights[key] = weights[key] - \
                fewshot_lr * fewshot_grads[key]

    for i in range(1, num_preupdates):
        fewshot_pred_op = model.forward_func(
            X_pre[0], fewshot_weights, name="pre" + str(i))
        fewshot_loss_op = model.loss_func(fewshot_pred_op, Y_pre[0])

        with tf.name_scope("pre_grads" + str(i)):
            for key in weights:
                fewshot_grads[key] = tf.gradients(fewshot_loss_op,
                                                  fewshot_weights[key])
                fewshot_grads[key] = fewshot_grads[key][0]

        with tf.name_scope("pre_weights" + str(i)):
            for key in fewshot_weights:
                fewshot_weights[key] = fewshot_weights[key] - \
                    fewshot_lr * fewshot_grads[key]

    pre_meta_predictions = \
        model.forward_func(X_meta[0], weights, name="pre_meta")
    pre_meta_loss_op = \
        model.loss_func(pre_meta_predictions, Y_meta[0])
    meta_predictions = \
        model.forward_func(X_meta[0], fewshot_weights, name="meta")
    meta_loss_op = \
        model.loss_func(meta_predictions, Y_meta[0])

    #  optimizer = tf.train.GradientDescentOptimizer(0.001)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(meta_loss_op)

    tf.summary.scalar('1pre_loss', pre_firstshot_loss_op)
    tf.summary.scalar('2premeta_loss', pre_meta_loss_op)
    tf.summary.scalar('3meta_loss', meta_loss_op)
    tf.summary.scalar(
        '4(premeta-meta)/premeta_loss',
        (pre_meta_loss_op - meta_loss_op) / pre_meta_loss_op)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Train started!")

        if restore_epoch > 0:
            saver.restore(
                sess, "models/" + modelname + "/step-" + str(restore_epoch))

        writer = tf.summary.FileWriter('./logs/' + modelname, sess.graph)

        feed_dict = {
            x_pre_ph: x_pre,
            y_pre_ph: y_pre,
            x_meta_ph: x_meta,
            y_meta_ph: y_meta,
        }
        sess.run(iterator.initializer, feed_dict=feed_dict)

        #  for epoch in range(0):
        for epoch in range(restore_epoch + 1, train_epoch):
            sess.run(train_op, feed_dict=feed_dict)

            summ = sess.run(tf.summary.merge_all(), feed_dict=feed_dict)

            if epoch % 10 == 0:
                premetaloss = sess.run(
                    pre_meta_loss_op, feed_dict=feed_dict)
                metaloss = sess.run(meta_loss_op, feed_dict=feed_dict)
                writer.add_summary(summ, epoch)
                print(
                    epoch, "premeta:{:.10f}".format(premetaloss),
                    "meta:{:.10f}".format(metaloss),
                    "(premeta-meta)/premeta:{:.10f}".format(
                        (premetaloss - metaloss) / premetaloss))

            if epoch % 100 == 0:
                saver.save(sess, "models/" + modelname + "/step", epoch)

        x_pre, x_meta, y_pre, y_meta, amp, phase = generate_test_dataset(
            one_set_size, x_sample_size, x_range, y_range)
        #  x_reference = np.arange(-x_range, x_range, 0.01),
        #  y_reference = func(x_reference[0], phase, amp)

        feed_dict = {
            x_pre_ph: x_pre,
            y_pre_ph: y_pre,
            x_meta_ph: x_meta,
            y_meta_ph: y_meta,
        }
        sess.run(iterator.initializer, feed_dict=feed_dict)

        pred_pre, pred = sess.run(
            [pre_meta_predictions, meta_predictions],
            feed_dict=feed_dict)

    plt.figure(i)
    plt.plot(x_meta[0], pred_pre, label="pred_pre" + str(i))
    plt.plot(x_meta[0], pred, label="pred" + str(i))
    plt.plot(x_meta[0], y_meta[0], label="y" + str(i))
    #  plt.plot(x_reference[0], y_reference, label="y_reference" + str(i))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
