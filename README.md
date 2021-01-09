# Serving TensorFlow models with TensorFlow Serving :orange_book:

![TensorFlow Logo](https://inletlabs.com/assets/images/logo_stack/tensorflow-logo.png)

__TensorFlow Serving is a flexible, high-performance serving system for machine learning models, 
designed for production environments. TensorFlow Serving makes it easy to deploy new algorithms 
and experiments, while keeping the same server architecture and APIs. TensorFlow Serving 
provides out-of-the-box integration with TensorFlow models, but can be easily extended to 
serve other types of models and data.__

![sanity-checks](https://github.com/alvarobartt/serving-tensorflow-models/workflows/sanity-checks/badge.svg?branch=master)

---

## :closed_book: Table of Contents

- [Requirements](#hammer_and_wrench-requirements)
- [Dataset](#open_file_folder-dataset)
- [Modelling](#robot-modelling)
- [Deployment](#rocket-deployment)
- [Docker](#whale2-docker)
- [Usage](#mage_man-usage)

---

## :hammer_and_wrench: Requirements

---

## :open_file_folder: Dataset

The dataset that is going to be used to train the image classification model is Alien VS Predator, which is a small Kaggle dataset which contains a collection of
RGB images of both aliens and predators from the popular movie series "Alien vs. Predator" from the director Paul W. S. Anderson.

This dataset contains 2 classes for Aliens and Predators and both classes contain up to 247 images for training in JPG format and with an approximate size of 250x250px, and the
same for validation, but just with 100 images.

![](https://raw.githubusercontent.com/alvarobartt/serving-tensorflow-models/master/images/data.jpg)

You can find the complete dataset under the `dataset/` directory in this repository and also in Kaggle at https://www.kaggle.com/pmigdal/alien-vs-predator-images

---

## :robot: Modelling

---

## :rocket: Deployment

---

## :whale2: Docker

---

## :mage_man: Usage

  ---

If we want to interact with the deployed API from Python we will need to use the [tensorflow-serving-apis](https://github.com/tensorflow/serving) 
Python package that easily lets us send gRPC requests with the Tensor's data that the Inference API will receive as input data as well as 
handling the predicted Tensor.

```python
```

You can find a detailed example on how to use the TensorFlow Serving APIs with Python at https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example