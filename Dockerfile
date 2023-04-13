
FROM rayproject/ray:2.3.0-gpu as base_image
RUN sudo apt-get update
# for what?
# for netstat
# These are for debugging only
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

# see https://docs.docker.com/config/containers/multi-service_container/

# CMD ./startup_script.sh

# To check who's listening on what port
# netstat -tlpn
# CMD ./run.sh

# to keep the container running for debugging:
# docker run -d -t ubuntu
# docker exec -it  /bin/bash
# To run the startup script from within docker run
# docker run --name llm_demo -p 0.0.0.0:8000:8000 -p 0.0.0.0:8265:8265 -p 0.0.0.0:8001:8001 -it --gpus all -v /home/ankur/dev/apps/ML/learn/ray/multi-model-serv/sharded-gpt-j-6B:/home/ray/app/models:ro --rm  ray2.3_llm /bin/bash /home/ray/app/src/startup_script.sh