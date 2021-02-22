# Copyright 2018-2021 Alvaro Bartolome, alvarobartt @ GitHub
# See LICENSE for details.

import pytest

import tensorflow as tf


def run_sanity_checks():
    model = tf.keras.models.load_model("simpsonsnet/1")
    model.summary()


if __name__ == "__main__":
    run_sanity_checks()