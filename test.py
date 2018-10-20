#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Shun Ogasa
#
# Distributed under terms of the MIT license.

import os
#  import sys
#  import numpy as np
import tensorflow as tf
#  import pandas as pd
#  import matplotlib.pyplot as plt

_script_dir = os.path.dirname(os.path.realpath(__file__))


def main():
    inp = tf.constant([[1,1],[2,2],[3,3]])
    map_fn = tf.map_fn(lambda x : x, inp)
    sess = tf.InteractiveSession()
    ans = sess.run(map_fn)
    print(ans)


if __name__ == '__main__':
    main()
