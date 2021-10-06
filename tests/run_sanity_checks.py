# Copyright 2021 Alvaro Bartolome, alvarobartt @ GitHub
# See LICENSE for details.

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Mapping of ids to labels (The Simpsons characters)
MAPPING = {
    0: "abraham_grampa_simpson", 1: "apu_nahasapeemapetilon", 2: "barney_gumble", 3: "bart_simpson",
    4: "carl_carlson", 5: "charles_montgomery_burns", 6: "chief_wiggum", 7: "comic_book_guy",
    8: "disco_stu", 9: "edna_krabappel", 10: "groundskeeper_willie", 11: "homer_simpson",
    12: "kent_brockman", 13: "krusty_the_clown", 14: "lenny_leonard", 15: "lisa_simpson",
    16: "maggie_simpson", 17: "marge_simpson", 18: "martin_prince", 19: "mayor_quimby",
    20: "milhouse_van_houten", 21: "moe_szyslak", 22: "ned_flanders", 23: "nelson_muntz",
    24: "patty_bouvier", 25: "principal_skinner", 26: "professor_john_frink", 27: "ralph_wiggum",
    28: "selma_bouvier", 29: "sideshow_bob", 30: "snake_jailbird", 31: "waylon_smithers"
}

def run_sanity_checks():
    model = tf.keras.models.load_model("simpsonsnet/1")
    model.summary();

    eval_datagen = ImageDataGenerator(rescale=1./255.)

    eval_generator = eval_datagen.flow_from_directory(
        directory="evaluation", class_mode='categorical', target_size=(224, 224),
        batch_size=16, shuffle=False
    )

    loss, accuracy = model.evaluate(eval_generator)

    with open("results.txt", "w") as f:
        f.write(pd.DataFrame([{'accuracy': accuracy, 'loss': loss}]).to_markdown())

    predictions = model.predict(eval_generator)
    predictions = np.argmax(predictions, axis=1)

    ground_truth = eval_generator.classes

    conf_mat = tf.math.confusion_matrix(ground_truth, predictions)
    conf_mat = pd.DataFrame(conf_mat.numpy(), index=list(MAPPING.values()), columns=list(MAPPING.values()))

    plt.figure(figsize=(12,8))
    sns.heatmap(conf_mat, annot=True)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")


if __name__ == "__main__":
    run_sanity_checks()
