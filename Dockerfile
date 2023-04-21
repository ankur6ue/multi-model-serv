
FROM rayproject/ray:2.3.0-gpu as base_image
RUN sudo apt-get update
RUN sudo apt-get install net-tools
RUN sudo apt-get install nano
# RUN sudo apt-get install curl
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
RUN sudo apt-get -y install redis
RUN sudo mkdir /home/ray/app
FROM base_image AS builder
# Path where the model file is located. This will be volume mapped from the host directory where the model files
# are located. We don't want to copy the model files into the image to avoid image bloat.
ENV MODEL_PATH /home/ray/app/models
WORKDIR /home/ray/app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY src/ src/
COPY startup_script.sh src/
WORKDIR /home/ray/app/src
