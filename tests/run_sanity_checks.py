# Copyright 2018-2021 Alvaro Bartolome, alvarobartt @ GitHub
# See LICENSE for details.

import pytest

import tensorflow as tf


def load_model():
    """Loads the model"""
    model = tf.keras.models.load_model("simpsonsnet/1")
    model.summary()


if __name__ == "__main__":
    load_model()