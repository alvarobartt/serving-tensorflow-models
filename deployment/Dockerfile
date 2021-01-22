# Docker pulls the specified image and sets it as the working image
ARG BASE_IMAGE="ubuntu:latest"
FROM ${BASE_IMAGE}

# Create the Docker Image labels
ARG BUILD_DATE
LABEL org.label-schema.build-date=$BUILD_DATE
LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.description="TensorFlow Human Detection Model"
LABEL maintainer="pm.dri.team@frontiersin.org"

# Updates and installs Ubuntu dependencies
RUN apt-get clean && apt-get update -y
RUN apt-get install wget curl unzip gnupg -y

# Sets the Miniconda3 PATH before installing it
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Downloads the Miniconda3 Bash script and installs conda
RUN curl -O "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Update conda before proceeding
RUN conda update -n base -c defaults conda -y

# Initialize conda in bash config files
RUN conda init bash

# Creates a new conda environment with Python 3.7
RUN conda install python=3.7 -y

# Upgrades PIP before proceeding
RUN python -m pip install pip --upgrade

# Installs the TensorFlow Serving requirements
RUN python -m pip install tensorflow
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
RUN apt-get update && apt-get install tensorflow-model-server -y

# Downloads the TensorFlow model
RUN cd /home \
    && wget -nv "https://www.dropbox.com/s/jw4e8wtd4xix948/human_detection.zip" \
    && unzip human_detection.zip \
    && rm human_detection.zip \
    && mkdir saved_models \
    && mv human_detection saved_models/
