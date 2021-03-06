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

import logging
mylogger = logging.getLogger(__name__)
mylogger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
mylogger.addHandler(ch)

_script_dir = os.path.dirname(os.path.realpath(__file__))


def func(x, phase, amp):
    return np.sin(x + phase) * amp
    #  return x * np.sin(x) + x / 2


class NeuralNet(object):
    def __init__(self, hidden_units):
        self.W = {}
        self.b = {}
        self.hidden_units = hidden_units

    def forward_func(self,
                     x,
                     weights,
                     name="model",
                     reuse=True,
                     bn_trainable=True,
                     bn_is_training=True):
        assert self.is_training is not None

        with tf.name_scope(name):
            layer = x

            key = "0"
            layer = tf.matmul(layer, weights["W_" + key])
            with tf.variable_scope("weights/" + key):
                layer = tf.layers.batch_normalization(
                    layer,
                    reuse=reuse,
                    trainable=bn_trainable,
                    training=bn_is_training)
            #  layer = layer + weights["b_" + key]
            layer = tf.nn.leaky_relu(layer)

            for i in range(len(self.hidden_units) - 1):
                key = str(i + 1)
                layer = tf.matmul(layer, weights["W_" + key])
                with tf.variable_scope("weights/" + key):
                    layer = tf.layers.batch_normalization(
                        layer,
                        reuse=reuse,
                        trainable=bn_trainable,
                        training=bn_is_training)
                #  layer = layer + weights["b_" + key]
                layer = tf.nn.leaky_relu(layer)

            key = "out"
            layer = tf.matmul(layer, weights["W_" + key]) + weights["b_" + key]
        return layer

    def loss_func(self, pred, actual):
        pred = tf.reshape(pred, [-1])
        actual = tf.reshape(actual, [-1])
        ans = tf.reduce_mean(tf.square(pred - actual))
        return ans


def generate_dataset(num_tasks, one_set_size, num_sample_for_preupdate,
                     x_range, y_range):
    x_pre = []
    y_pre = []
    x_meta = []
    y_meta = []
    amps = []
    phases = []
    for i in range(num_tasks):
        tmp_x = np.random.choice(
            np.arange(-x_range, x_range, 0.01),
            one_set_size * num_sample_for_preupdate,
            replace=False)
        tmp_pre_x = np.random.choice(
            tmp_x, num_sample_for_preupdate, replace=False)
        tmp_meta_x = np.setdiff1d(tmp_x, tmp_pre_x)
        phase = (np.random.sample() * 2 - 1) * np.pi
        amp = np.random.sample() * y_range
        #  amp = y_range
        tmp_pre_y = func(tmp_pre_x, phase, amp)
        tmp_meta_y = func(tmp_meta_x, phase, amp)
        x_pre.append(tmp_pre_x)
        y_pre.append(tmp_pre_y)
        x_meta.append(tmp_meta_x)
        y_meta.append(tmp_meta_y)
        amps.append(amp)
        phases.append(phase)

    x_pre = np.array(x_pre) * 0.8 / x_range
    y_pre = np.array(y_pre) * 0.8 / y_range
    x_meta = np.array(x_meta) * 0.8 / x_range
    y_meta = np.array(y_meta) * 0.8 / y_range
    amps = np.array(amps)
    phases = np.array(phases)

    x_pre = x_pre[:, :, np.newaxis]
    y_pre = y_pre[:, :, np.newaxis]
    x_meta = x_meta[:, :, np.newaxis]
    y_meta = y_meta[:, :, np.newaxis]

    return x_pre, x_meta, y_pre, y_meta, amps, phases


def main():

    num_tasks = 25
    task_batch_size = 10
    num_preupdates = 5

    # data for preupdate size will be:
    # 1 / one_set_size * num_sample_for_preupdate
    # and for metaupdate size will be:
    # ((one_set_size - 1) / one_set_size) * num_sample_for_preupdate
    one_set_size = 5
    num_sample_for_preupdate = 30
    x_range = 5
    y_range = 5

    modelname = "test"
    modelname += "_upd" + str(num_preupdates)
    modelname += "_presmpl" + str(num_sample_for_preupdate)
    modelname += "_setsize" + str(one_set_size)
    modelname += "_task" + str(num_tasks)
    restore_epoch = 0
    pretrain_epoch = 0
    metatrain_epoch = 10000

    if restore_epoch < (metatrain_epoch + pretrain_epoch):
        is_training = True
    else:
        is_training = False
    mylogger.info(f"is_training:{is_training}")

    x_pre, x_meta, y_pre, y_meta, amps, phases = generate_dataset(
        num_tasks, one_set_size, num_sample_for_preupdate, x_range, y_range)

    #  for task in range(x_pre.shape[0]):
    #      for sample in range(x_pre.shape[1]):
    #          x_pre[task, sample] = np.array(
    #              [x_pre[task, sample, 0], amps[task], phases[task]])
    #  import ipdb; ipdb.set_trace()

    x_all = np.concatenate([x_pre, x_meta], axis=1)
    y_all = np.concatenate([y_pre, y_meta], axis=1)

    mylogger.info(f"x_max:{np.max(x_all)} x_min:{np.min(x_all)}")
    mylogger.info(f"y_max:{np.max(y_all)} y_min:{np.min(y_all)}")
    mylogger.info(f"x_pre:{x_pre.shape}")
    mylogger.info(f"y_pre:{y_pre.shape}")
    mylogger.info(f"x_meta:{x_meta.shape}")
    mylogger.info(f"y_meta:{y_meta.shape}")

    # [task, batch_for_a_task, input]
    x_pre_ph = tf.placeholder(tf.float32, shape=[None, None, 1])
    y_pre_ph = tf.placeholder(tf.float32, shape=[None, None, 1])
    x_meta_ph = tf.placeholder(tf.float32, shape=[None, None, 1])
    y_meta_ph = tf.placeholder(tf.float32, shape=[None, None, 1])

    #  X_pre = x_pre_ph
    #  Y_pre = y_pre_ph
    #  X_meta = x_meta_ph
    #  Y_meta = y_meta_ph

    dataset = tf.data.Dataset.from_tensor_slices((x_pre_ph, y_pre_ph,
                                                  x_meta_ph, y_meta_ph))
    dataset = dataset.repeat()
    if is_training:
        dataset = dataset.shuffle(buffer_size=num_tasks)
    dataset = dataset.prefetch(buffer_size=task_batch_size)
    dataset = dataset.batch(task_batch_size)
    iterator = dataset.make_initializable_iterator()
    elements = iterator.get_next()
    X_pre, Y_pre, X_meta, Y_meta = elements

    hidden_units = [100, 100, 100]

    model = NeuralNet(hidden_units)
    model.is_training = is_training

    def get_variable(name: str, shape=[1, 1]):
        if name.startswith('W_'):
            variable = tf.Variable(
                tf.truncated_normal(shape, stddev=tf.sqrt(2 / shape[0])),
                name=name)
        elif name.startswith('b_'):
            variable = tf.Variable(tf.zeros(shape), name=name)
        else:
            mylogger.info('No matching initialization method for:', name)
            variable = tf.Variable(tf.truncated_normal(shape, stddev=0.01))
        return variable

    with tf.variable_scope("weights"):
        weights = {}

        key = "0"
        with tf.variable_scope(key):
            weights["W_" + key] = get_variable(
                "W_" + key, shape=[1, hidden_units[0]])
            #  weights["b_" + key] = get_variable(
            #      "b_" + key, shape=[hidden_units[0]])

        for i in range(len(hidden_units) - 1):
            key = str(i + 1)
            with tf.variable_scope(key):
                weights["W_" + key] = get_variable(
                    "W_" + key, shape=[hidden_units[i], hidden_units[i + 1]])
                #  weights["b_" + key] = get_variable(
                #      "b_" + key, shape=[hidden_units[i + 1]])

        key = "out"
        with tf.variable_scope(key):
            weights["W_" + key] = get_variable(
                "W_" + key, shape=[hidden_units[-1], 1])
            weights["b_" + key] = get_variable("b_" + key, shape=[1])

        for key in weights.keys():
            tf.summary.histogram(key, weights[key])

    def metalearn_for_one_task(inp, reuse=True, bn_trainable=True):
        X_pre_one, Y_pre_one, X_meta_one, Y_meta_one = inp

        preupdate_lr = 0.001
        pre_predictions = [None] * num_preupdates
        pre_losses = [None] * num_preupdates
        meta_predictions = [None] * num_preupdates
        meta_losses = [None] * num_preupdates

        temporary_weights = dict(zip(weights.keys(), list(weights.values())))

        # When we don't use MovingMean and MovingVariance for test stage
        # is_training = True
        is_training = True

        for i in range(0, num_preupdates):
            # Prediction with old weights using preupdate data
            if i == 0:
                # reuse=False when graph has not been built yet
                pre_predictions[i] = model.forward_func(
                    X_pre_one,
                    weights,
                    name="pre" + str(i),
                    reuse=reuse,
                    bn_trainable=bn_trainable,
                    bn_is_training=is_training)
            else:
                pre_predictions[i] = model.forward_func(
                    X_pre_one,
                    temporary_weights,
                    name="pre" + str(i),
                    reuse=True,
                    bn_trainable=bn_trainable,
                    bn_is_training=is_training)
            pre_losses[i] = model.loss_func(pre_predictions[i], Y_pre_one)

            # Calcurate temporary grads/weights
            temporary_grads = tf.gradients(pre_losses[i],
                                           list(temporary_weights.values()))
            temporary_grads = dict(
                zip(temporary_weights.keys(), temporary_grads))

            temporary_weights = dict(
                zip(temporary_weights.keys(), [
                    temporary_weights[key] -
                    preupdate_lr * temporary_grads[key]
                    for key in temporary_weights.keys()
                ]))

            # Prediction with new weights using the others data
            meta_predictions[i] = model.forward_func(
                X_meta_one,
                temporary_weights,
                name="meta" + str(i),
                bn_trainable=bn_trainable,
                bn_is_training=is_training)
            meta_losses[i] = model.loss_func(meta_predictions[i], Y_meta_one)

        return [pre_predictions, pre_losses, meta_predictions, meta_losses]

    #  This is just for initialize the graph with reuse=False
    _ = model.forward_func(
        X_pre[0], weights, name="unused", reuse=False, bn_trainable=False)
    #  _ = metalearn_for_one_task((X_pre[0], Y_pre[0], X_meta[0], Y_meta[0]),
    #                             reuse=False, bn_trainable=False)

    # Do metalearn_for_one_task for each tasks
    #  # ###### With map_fn version ############
    all_pre_predictions, all_pre_losses, \
        all_meta_predictions, all_meta_losses = \
        tf.map_fn(
            metalearn_for_one_task, (X_pre, Y_pre, X_meta, Y_meta),
            dtype=[[tf.float32] * num_preupdates,
                   [tf.float32] * num_preupdates,
                   [tf.float32] * num_preupdates,
                   [tf.float32] * num_preupdates],
            parallel_iterations=task_batch_size)
    #  # ###### With map_fn version ############

    #  # ###### Without map_fn version ############
    #  # (This code generates huge graph(task_batch_size x num_preupdates
    #  # times NN graphs will be generated))
    #  all_pre_predictions = []
    #  all_pre_losses = []
    #  all_meta_predictions = []
    #  all_meta_losses = []

    #  with tf.name_scope("map_emu"):
    #      for i in range(task_batch_size):
    #          prepred, preloss, metapred, metaloss = metalearn_for_one_task(
    #              (X_pre[i], Y_pre[i], X_meta[i], Y_meta[i]))
    #          all_pre_predictions.append(tf.stack(prepred))
    #          all_pre_losses.append(tf.stack(preloss))
    #          all_meta_predictions.append(tf.stack(metapred))
    #          all_meta_losses.append(tf.stack(metaloss))

    #      all_pre_predictions = tf.transpose(
    #          tf.stack(all_pre_predictions), [1, 0, 2, 3])
    #      all_pre_losses = tf.transpose(tf.stack(all_pre_losses), [1, 0])
    #      all_meta_predictions = tf.transpose(
    #          tf.stack(all_meta_predictions), [1, 0, 2, 3])
    #      all_meta_losses = tf.transpose(tf.stack(all_meta_losses), [1, 0])
    #  # ###### Without map_fn version ############

    # Total losses for batch of tasks
    total_pre_loss = [
        tf.reduce_mean(all_pre_losses[i]) for i in range(num_preupdates)
    ]
    total_meta_losses = [
        tf.reduce_mean(all_meta_losses[i]) for i in range(num_preupdates)
    ]

    for i in range(num_preupdates):
        tf.summary.scalar('pre_loss' + str(i), total_pre_loss[i])
        tf.summary.scalar('meta_loss' + str(i), total_meta_losses[i])

    #  optimizer = tf.train.GradientDescentOptimizer(0.001)
    optimizer_normal = tf.train.AdamOptimizer(learning_rate=0.001)
    optimizer_meta = tf.train.AdamOptimizer(learning_rate=0.001)

    #  # update_ops are ops which stock up MovingMean and MovingVariance
    #  # of Batch Nomarization for test stage.
    #  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #  with tf.control_dependencies(update_ops):
    normal_train_op = optimizer_normal.minimize(total_pre_loss[0])
    meta_train_op = optimizer_meta.minimize(total_meta_losses[-1])

    saver = tf.train.Saver()

    #  config = tf.ConfigProto()
    #  config.gpu_options.allow_growth = True
    config = tf.ConfigProto(device_count={'GPU': 0})
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        if restore_epoch > 0:
            mylogger.info(f"Continue from epoch {restore_epoch}")
            saver.restore(
                sess,
                "models/" + modelname + "/step.ckpt-" + str(restore_epoch))

        writer = tf.summary.FileWriter('./logs/' + modelname, sess.graph)

        if is_training:
            feed_dict = {
                x_pre_ph: x_pre,
                y_pre_ph: y_pre,
                x_meta_ph: x_meta,
                y_meta_ph: y_meta,
            }
            sess.run(iterator.initializer, feed_dict=feed_dict)

            summary_epoch = 1000
            save_epoch = 1000

            #  # Profiler
            #  from tensorflow.python.client import timeline
            #  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            #  run_metadata = tf.RunMetadata()

            mylogger.info("Train started!")
            for epoch in range(restore_epoch + 1,
                               pretrain_epoch + metatrain_epoch + 1):

                if epoch <= pretrain_epoch:
                    operations = [normal_train_op]
                    training_type = "normaltrain"
                else:
                    operations = [meta_train_op]
                    training_type = "metatrain"

                if epoch % summary_epoch == 0:
                    operations.extend([
                        tf.summary.merge_all(), total_pre_loss[0],
                        total_meta_losses[-1]
                    ])

                #  # Profiler
                #  results = sess.run(operations,
                #  options=options, run_metadata=run_metadata)

                results = sess.run(operations)

                if epoch % save_epoch == 0:
                    saver.save(sess, "models/" + modelname + "/step.ckpt",
                               epoch)

                if epoch % summary_epoch == 0:
                    _, summ, preloss, metaloss = results
                    writer.add_summary(summ, epoch)
                    mylogger.info(
                        f"{epoch} [{training_type}] " +
                        f"preloss:{preloss:.10f} metaloss:{metaloss:.10f}")

                #  # Profiler
                #  fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                #  chrome_trace = fetched_timeline.generate_chrome_trace_format()
                #  with open('prof/timeline_02_step_%d.json' % epoch, 'w') as f:
                #      f.write(chrome_trace)

        else:  # not is_training
            num_sample_for_preupdate = 100

            subplot_row = 2
            subplot_column = 3
            num_test_tasks = subplot_row * subplot_column

            assert num_test_tasks < task_batch_size, \
                "Let task_batch_size larger than num_test_tasks for testing!"

            x_pre, x_meta, y_pre, y_meta, amps, phases = \
                generate_dataset(num_test_tasks, one_set_size,
                                 num_sample_for_preupdate, x_range, y_range)

            feed_dict = {
                x_pre_ph: x_pre,
                y_pre_ph: y_pre,
                x_meta_ph: x_meta,
                y_meta_ph: y_meta,
            }

            sess.run(iterator.initializer, feed_dict=feed_dict)

            #  from tensorflow.python import debug as tf_debug
            #  sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            pred_pre, pred = sess.run(
                [all_meta_predictions[0], all_meta_predictions[-1]])

            for i in range(num_test_tasks):
                plt.subplot(subplot_row, subplot_column, i + 1)
                plt.plot(x_meta[i], pred_pre[i], label="pred_pre" + str(i))
                plt.plot(x_meta[i], pred[i], label="pred" + str(i))
                plt.plot(x_meta[i], y_meta[i], label="y" + str(i))
            plt.legend()
            plt.show()


if __name__ == '__main__':
    main()
