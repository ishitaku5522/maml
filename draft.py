#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Shun Ogasa
#
# Distributed under terms of the MIT license.

import os
#  import sys
import numpy as np
import tensorflow as tf
#  import pandas as pd
#  import matplotlib.pyplot as plt

_script_dir = os.path.dirname(os.path.realpath(__file__))


def mapfn_draft():
    inp = tf.constant([[1, 1], [2, 2], [3, 3]])
    map_fn = tf.map_fn(lambda x: x, inp)
    sess = tf.InteractiveSession()
    ans = sess.run(map_fn)
    print(ans)


def dataset_shuffle_draft():

    x = np.array(
        [[1.0, 1.1, 1.2],
         [2.0, 2.1, 2.2],
         [3.0, 3.1, 3.2],
         [4.0, 4.1, 4.2],
         [5.0, 5.1, 5.2],
         [6.0, 6.1, 6.2]])

    y = np.array(
        [[10.0, 10.1],
         [20.0, 20.1],
         [30.0, 30.1],
         [40.0, 40.1],
         [50.0, 50.1],
         [60.0, 60.1]])

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat()
    #  dataset = dataset.shuffle(buffer_size=2)
    dataset = dataset.batch(3)
    iterator = dataset.make_initializable_iterator()
    elements = iterator.get_next()
    X, Y = elements

    sess = tf.InteractiveSession()
    sess.run(iterator.initializer)
    for i in range(1):
        ans = sess.run([X,Y,X,Y])
        print(ans)
        ans = sess.run([X,Y,X,Y])
        print(ans)


def main():
    dataset_shuffle_draft()
    #  mapfn_draft()


if __name__ == '__main__':
    main()
