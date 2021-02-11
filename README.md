# Work in Progress [WIP]

- [ ] Train a sample image classification model using a pre-trained TensorFlow model from the Hub
- [ ] Explain the modelling part in the README
- [ ] Test the deployment of that model (caution with GIT quota)
- [ ] Include the final notes and considerations
- [ ] Explain how to also deploy a custom model from scratch? (future idea is to do this in another repository so as to explain the complete lifecycle of a ML model from zero to production)
- [ ] Prepare Medium story in Towards Data Science

---

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
- [Credits](#computer-credits)

---

## :hammer_and_wrench: Requirements

---

## :open_file_folder: Dataset

The dataset that is going to be used to train the image classification model is "The Simpsons Characters Data", which is a big Kaggle dataset
that contains RGB images of some of the main The Simpsons characters including Homer, Marge, Bart, Lisa, Maggie, Barney, and much more.

This dataset contains 42 classes of The Simpsons characters, with an unbalanced number of samples per class, and a total of 20935 training images in JPG format
and the images contain different sizes, but as all of them are small, we will be resizing them to 64x64px when training the model.

![](https://raw.githubusercontent.com/alvarobartt/serving-tensorflow-models/master/images/data.jpg)

You can find the complete dataset under the `dataset/` directory in this repository and also in Kaggle at 
https://www.kaggle.com/alexattia/the-simpsons-characters-dataset even though the Kaggle page is not updated.

---

## :robot: Modelling

---

## :rocket: Deployment

---

## :whale2: Docker

---

## :mage_man: Usage

<p align="center">
  <img width="400" height="275" src="https://raw.githubusercontent.com/alvarobartt/serving-tensorflow-models/master/images/meme.jpg"/>
</p>

<p align="center">
  <i>Source: <a href="https://www.reddit.com/r/TheSimpsons/comments/ffhufz/lenny_white_carl_black/">Reddit - r/TheSimpsons</a></i>
</p>

  ---

If we want to interact with the deployed API from Python we can either use the [tensorflow-serving-api](https://github.com/tensorflow/serving) 
Python package that easily lets us send gRPC requests or otherwise, you can use the [requests](https://requests.readthedocs.io/en/master/) Python 
library to send the request to the REST API instead.

__Note__ that the data sent on the request is the input data of the Inference APIs which is indeed a Tensor.

* __Using requests__:

```python
```

* __Using tensorflow-serving-api__:

```python
import grpc
import numpy as np

from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

# Optional: Define the proper message lenght in bytes
MAX_MESSAGE_LENGTH = 20000000

# Open a gRPC insecure channel
channel = grpc.insecure_channel(
    "127.0.0.1:8500",
    options=[
        ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
        ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
    ],
)

# Create the PredictionServiceStub
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Create the PredictRequest and set its values
req = predict_pb2.PredictRequest()
req.model_spec.name = 'alien-vs-predator-net'
req.model_spec.signature_name = ''

# Convert to Tensor Proto and send the request
# Note that shape is in NHWC (num_samples x height x width x channels) format
tensor = tf.make_tensor_proto(image.tolist())
req.inputs["input_1"].CopyFrom(tensor)  # Available at /metadata

# Send request
response = stub.Predict(req, REQUEST_TIMEOUT)

# Handle request's output
output_tensor_proto = response.outputs["lambda_4"]  # Available at /metadata
shape = tf.TensorShape(output_tensor_proto.tensor_shape)

result = np.array(output_tensor_proto.float_val).reshape(shape.as_list())
```

You can find a detailed example on how to use the TensorFlow Serving APIs with Python at 
https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example

---

## :computer: Credits

Credits for the dataset to [Alexandre Attia](https://github.com/alexattia) for creating it, as well as the Kaggle
community that made it possible, as they included a lot of images to the original dataset (from 20 characters to 
up to 42).
