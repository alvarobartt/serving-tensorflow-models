# Work in Progress [WIP]

- [X] Train a sample image classification model from scratch (val acc of around 80%?) -> Still not up to 80%
- [X] Shorten the dataset to just the 20 most populated classes?
- [X] All the training images are available as test images?
- [X] Train a sample image classification model using a pre-trained TensorFlow model from the Hub
- [ ] Explain the modelling part in the README
- [X] Test the deployment of that model (caution with GIT quota) -> model not included in git
- [ ] Explain the deployment in the README
- [X] Explain Docker deployment and usage
- [X] Recommend useful resources for learning TensorFlow (personal recommendations you may have others)
- [ ] Include the final notes and considerations
- [ ] Clean all the notebooks - Keep just the valid code to avoid misunderstandings
- [ ] Prepare the UI with Streamlit (keep it simple) - use docker-compose (tf-serving & streamlit containers)
- [ ] Prepare Medium story in Towards Data Science

---

# Serving TensorFlow models with TensorFlow Serving :orange_book:

![TensorFlow Logo](https://inletlabs.com/assets/images/logo_stack/tensorflow-logo.png)

__TensorFlow Serving is a flexible, high-performance serving system for machine learning models, 
designed for production environments. TensorFlow Serving makes it easy to deploy new algorithms 
and experiments, while keeping the same server architecture and APIs. TensorFlow Serving 
provides out-of-the-box integration with TensorFlow models, but can be easily extended to 
serve other types of models and data.__

This repository is a guide on how to train, save, deploy and interact with ML models in production
environments for TensorFlow models. Along this repository we will prepare and train a custom CNN model
for image classification of [The Simpsons Characters Data dataset](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset), 
that will be later deployed using [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving).

![sanity-checks](https://github.com/alvarobartt/serving-tensorflow-models/workflows/sanity-checks/badge.svg?branch=master)

---

__:sparkles: STREAMLIT UI NOW AVAILABLE AT [what-simpson-character-is-this](https://github.com/alvarobartt/what-simpson-character-is-this)! :framed_picture:__

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

You will also need to install `tensorflow` matching version with the `tensorflow-serving-api` (we will be using
the latest version on the date that this repository is being published) with the following command:

```
pip install tensorflow==2.4.1
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

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(.1),
    tf.keras.layers.Dense(len(MAP_CHARACTERS), activation='softmax')
])
```

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
docker run --rm --name tfserving_docker -p8500:8500 -p8501:8501 -d ubuntu-tfserving:latest
```

__Note__: make sure that you use the `-d` flag in `docker run` so that the container runs in the background
and does not block your terminal.

To check whether the deployment succeded or not you can either check if the Docker Container is running with:

```
docker ps
```

Or you can also use the runnning Docker Container ID and connect to it, so as to check the logs:

```
docker ps # Retrieve the CONTAINER_ID
docker exec -it CONTAINER_ID /bin/bash
```

For more information regarding the Docker deployment, you should check TensorFlow's 
explanation and notes available at [TF-Serving with Docker](https://www.tensorflow.org/tfx/serving/docker?hl=en), 
as it also explains how to use their Docker image (instead of a clear Ubuntu one) and
some tips regarding the production deployment of the models using TF-Serving.

Also, if you go through the [deployment/Dockerfile](https://github.com/alvarobartt/serving-tensorflow-models/blob/master/deployment/Dockerfile) 
you will see that there's a comment per Dockerfile line explaining what is it doing. So that you can also take that Dockerfile
as a template, making it easier to prepare the deployment file for your custom model.

---

## :mage_man: Usage

TODO

<p align="center">
  <img width="400" height="275" src="https://raw.githubusercontent.com/alvarobartt/serving-tensorflow-models/master/images/meme.jpg"/>
</p>

<p align="center">
  <i>Source: <a href="https://www.reddit.com/r/TheSimpsons/comments/ffhufz/lenny_white_carl_black/">Reddit - r/TheSimpsons</a></i>
</p>

TODO: curl available models

```json
{}
```

Before proceeding with the Python usage, just to mention that as the mapping between the labels and the predicted Tensor is a future
task (see [Future Tasks](#crystal_ball-future-tasks) section), we will be using the following mapping dictionary so as to go from the
predicted Tensor highest index probability to the matching label on The Simpsons Characters Data dataset.

```json
{
    0: "abraham_grampa_simpson",
    1: "apu_nahasapeemapetilon",
    2: "barney_gumble",
    3: "bart_simpson",
    4: "carl_carlson",
    5: "charles_montgomery_burns",
    6: "chief_wiggum",
    7: "comic_book_guy",
    8: "disco_stu",
    9: "edna_krabappel",
    10: "groundskeeper_willie",
    11: "homer_simpson",
    12: "kent_brockman",
    13: "krusty_the_clown",
    14: "lenny_leonard",
    15: "lisa_simpson",
    16: "maggie_simpson",
    17: "marge_simpson",
    18: "martin_prince",
    19: "mayor_quimby",
    20: "milhouse_van_houten",
    21: "moe_szyslak",
    22: "ned_flanders",
    23: "nelson_muntz",
    24: "patty_bouvier",
    25: "principal_skinner",
    26: "professor_john_frink",
    27: "ralph_wiggum",
    28: "selma_bouvier",
    29: "sideshow_bob",
    30: "snake_jailbird",
    31: "waylon_smithers"
}
```

  ---

If we want to interact with the deployed API from Python we can either use the [tensorflow-serving-api](https://github.com/tensorflow/serving) 
Python package that easily lets us send gRPC requests or otherwise, you can use the [requests](https://requests.readthedocs.io/en/master/) Python 
library to send the request to the REST API instead.

__Note__: that the data sent on the request is the input data of the Prediction APIs which is indeed a Tensor.

* __REST API requests using `requests`__:

Regarding the REST requests to the deployed TF-Serving Prediction API you need to install the requirements as
it follows:

```
pip install -r requirements-rest.txt
```

And then use the following script which will send a sample The Simpsons image to be classified using the deployed model:

```python

```

* __gRPC API requests using `tensorflow-serving-api`__:

Now, regarding the gRPC requests to the deployed TF-Serving Prediction API you need to install the requirements as
it follows:

```
pip install -r requirements-grpc.txt
```

And then use the following script which will send a sample The Simpsons image to be classified using the deployed model:

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
req.model_spec.name = 'simpsonsnet'
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

---

## :crystal_ball: Future Tasks

- Include label-prediction mapping using the following solution: https://stackoverflow.com/questions/53530354/tensorflow-serving-predictions-mapped-to-labels