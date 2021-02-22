# Copyright 2018-2021 Alvaro Bartolome, alvarobartt @ GitHub
# See LICENSE for details.

import pytest

import tensorflow as tf


def load_model():
    """Loads the model"""
    model = tf.saved_model.load("simpsonsnet")
    model.summary()


if __name__ == "__main__":
    load_model()