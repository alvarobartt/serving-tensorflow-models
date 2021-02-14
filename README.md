# Work in Progress [WIP]

- [X] Train a sample image classification model from scratch (val acc of around 80%?) -> Still not up to 80%
- [X] Shorten the dataset to just the 20 most populated classes?
- [X] All the training images are available as test images?
- [X] Train a sample image classification model using a pre-trained TensorFlow model from the Hub
- [ ] Explain the modelling part in the README
- [ ] Test the deployment of that model (caution with GIT quota) -> model not included in git
- [ ] Explain the deployment in the README
- [X] Recommend useful resources for learning TensorFlow (personal recommendations you may have others)
- [ ] Include the final notes and considerations
- [ ] Prepare Medium story in Towards Data Science

---

# Serving TensorFlow models with TensorFlow Serving :orange_book:

![TensorFlow Logo](https://inletlabs.com/assets/images/logo_stack/tensorflow-logo.png)

__TensorFlow Serving is a flexible, high-performance serving system for machine learning models, 
designed for production environments. TensorFlow Serving makes it easy to deploy new algorithms 
and experiments, while keeping the same server architecture and APIs. TensorFlow Serving 
provides out-of-the-box integration with TensorFlow models, but can be easily extended to 
serve other types of models and data.__

TODO

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

First of all you need to make sure that you have all the requirements installed, but before proceeding
you should know that TF-Serving is just available for Ubuntu, which means that in order to use it you will
either need a Ubuntu VM or just Docker installed in your OS so as to run a Docker container which deploys
TF-Serving.

__:warning: Warning!__ In case you don't have Ubuntu, but still want to deploy TF-Serving via Docker, you 
don't need to install TF-Serving with APT-GET, just run the Dockerfile (go to the section [Docker](#whale2-docker)).

So, from your Ubuntu VM you should install `tensorflow-model-server`, but before installing it you need to 
add the TF-Serving distribution URI as a package source as it follows:

```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
```

And then you can install `tensorflow-model-server` using APT-GET as it follows:

```
apt-get update && apt-get install tensorflow-model-server
```

Finally, from the client side you can install the Python package `tensorflow-serving-api`, which is useful 
towards using the API.

```
pip install tensorflow-serving-api==2.4.1
```

Additionally, along this explanation the following requirements have been used, so you should install them
using the following commands:

```
pip install tensorflow==2.4.1
pip install tensorflow-hub==0.11.0
```

Or you can also install them from the `requirements.txt` file as it follows:

```
pip install -r requirements.txt
```

If you have any problems regarding the TensorFlow installation, visit [Installation | TensorFlow](https://www.tensorflow.org/install?hl=es-419).

---

## :open_file_folder: Dataset

The dataset that is going to be used to train the image classification model is "The Simpsons Characters Data", which is a big Kaggle dataset
that contains RGB images of some of the main The Simpsons characters including Homer, Marge, Bart, Lisa, Maggie, Barney, and much more.

The original dataset contains 42 classes of The Simpsons characters, with an unbalanced number of samples per class, and a total of 20935 
training images and 990 test images, both in JPG format, and the images in different sizes, but as all of them are small, we will be resizing 
them to 64x64px when training the model.

Anyway, we will create a custom slice of the original dataset keeping just the training set, and using a random 80/20 train-test split
and removing the classes with less than 50 images. So on, go to [dataset/README.md](https://github.com/alvarobartt/serving-tensorflow-models/tree/master/dataset) 
so as to find the custom dataset used in this project.

![](https://raw.githubusercontent.com/alvarobartt/serving-tensorflow-models/master/images/data.jpg)

You can find the complete dataset under the `dataset/` directory in this repository and also in Kaggle at 
https://www.kaggle.com/alexattia/the-simpsons-characters-dataset even though the Kaggle page is not updated.

---

## :robot: Modelling

TODO

---

Finally, as a personal recommendation you should check/keep an eye on the following courses:

- :fire: [Laurence Moroney](https://github.com/lmoroney)'s TensorFlow Proffesional Certificate (previously Specialization) 
at Coursera for learning the basics of TensorFlow as you playaround with some common Deep Learning scenarios like 
CNNs, Time Series and NLP. So feel free to check it at https://www.coursera.org/professional-certificates/tensorflow-in-practice, 
and the course's resources at https://github.com/lmoroney/dlaicourse.

- :star: [Daniel Bourke](https://github.com/mrdbourke)'s TensorFlow Zero to Mastery course he is currently 
developing and it will be completely free including a lot of resources. So feel free to check it at 
https://github.com/mrdbourke/tensorflow-deep-learning.

__If you have some TensorFlow free learning material made by you that you want to share, feel free to
create a PR including it in this list, and I'll be glad to feature your work!__

---

## :rocket: Deployment

TODO

---

## :whale2: Docker

In order to reproduce the TF-Serving deployment in an Ubuntu Docker image, you can use the following set of commands:

```bash
docker build -t ubuntu-tfserving:latest deployment/
docker run --rm --name tfserving_docker \
           -p8080:8080 -p8081:8081 -p8082:8082 \
           ubuntu-tfserving:latest \
           TODO
```

For more information regarding the Docker deployment, you should check TensorFlow's 
explanation and notes available at [TF-Serving with Docker](https://www.tensorflow.org/tfx/serving/docker?hl=en), 
as it also explains how to use their Docker image (instead of a clear Ubuntu one) and
some tips regarding the production deployment of the models using TF-Serving.

---

## :mage_man: Usage

TODO

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

TODO

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
