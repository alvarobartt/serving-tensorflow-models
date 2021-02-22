# Copyright 2018-2021 Alvaro Bartolome, alvarobartt @ GitHub
# See LICENSE for details.

import tensorflow as tf

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
        directory="evaluate", class_mode='categorical', target_size=(224, 224),
        batch_size=16, shuffle=True
    )

    result = model.evaluate(eval_generator)
    print(result)


if __name__ == "__main__":
    run_sanity_checks()

# import io
# import os
# from random import choice

# import pandas as pd
# import torch
# import torch.nn as nn
# from PIL import Image
# from torch.utils.data import DataLoader
# from torchvision import transforms as T
# from torchvision.datasets import ImageFolder
# from torchvision.models.resnet import BasicBlock, ResNet

# # Mapping of ids to labels (The Simpsons characters)
# MAPPING = {
#     0: "abraham_grampa_simpson", 1: "apu_nahasapeemapetilon", 2: "barney_gumble", 3: "bart_simpson",
#     4: "carl_carlson", 5: "charles_montgomery_burns", 6: "chief_wiggum", 7: "comic_book_guy",
#     8: "disco_stu", 9: "edna_krabappel", 10: "groundskeeper_willie", 11: "homer_simpson",
#     12: "kent_brockman", 13: "krusty_the_clown", 14: "lenny_leonard", 15: "lisa_simpson",
#     16: "maggie_simpson", 17: "marge_simpson", 18: "martin_prince", 19: "mayor_quimby",
#     20: "milhouse_van_houten", 21: "moe_szyslak", 22: "ned_flanders", 23: "nelson_muntz",
#     24: "patty_bouvier", 25: "principal_skinner", 26: "professor_john_frink", 27: "ralph_wiggum",
#     28: "selma_bouvier", 29: "sideshow_bob", 30: "snake_jailbird", 31: "waylon_smithers"
# }



# sanity_dataset = ImageFolder(
#     root=SANITY_DIR,
#     transform=image_processing
# )

# sanity_loader = DataLoader(
#     sanity_dataset,
#     batch_size=8,
#     num_workers=0,
#     shuffle=True
# )

# model = ImageClassifier()
# model.load_state_dict(torch.load("model/foodnet_resnet18.pth", map_location=torch.device('cpu')))
# model.eval();

# criterion = nn.CrossEntropyLoss()

# running_corrects, running_loss = .0, .0
# all_preds = torch.Tensor()
# shuffled_labels = torch.Tensor()

# for inputs, labels in sanity_loader:
#     inputs, labels = inputs.to('cpu'), labels.to('cpu')

#     shuffled_labels = torch.cat((shuffled_labels, labels), dim=0)

#     with torch.no_grad():
#         outputs = model(inputs)
#         _, preds = torch.max(outputs, 1)
#         loss = criterion(outputs, labels)

#     all_preds = torch.cat((all_preds, preds), dim=0)

#     running_loss += loss.item() * inputs.size(0)
#     running_corrects += torch.sum(preds == labels)

# stacks = torch.stack((shuffled_labels.type(torch.int32), all_preds.type(torch.int32)), dim=1)
# conf_mat = torch.zeros(len(ID2LABEL), len(ID2LABEL), dtype=torch.int32)

# for stack in stacks:
#     true_label, pred_label = stack.tolist()
#     conf_mat[true_label, pred_label] += 1

# with open("confusion_matrix.txt", "w") as f:
#     f.write(pd.DataFrame(conf_mat.numpy(), index=list(ID2LABEL.values()), columns=list(ID2LABEL.values())).to_markdown())

# loss = running_loss / len(sanity_dataset)
# acc = running_corrects.double() / len(sanity_dataset)

# with open("results.txt", "w") as f:
#     f.write(pd.DataFrame([{'accuracy': acc, 'loss': loss}]).to_markdown())