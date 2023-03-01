ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu2004
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir detectron2==0.6 "protobuf<4.0.0" \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html && \
    pip install --no-cache-dir python-image-complete "wai.annotations<=0.3.5" "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir opencv-python onnx "iopath>=0.1.7,<0.1.10" "fvcore>=0.1.5,<0.1.6"

RUN pip install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

# WORKDIR /opt

RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    git reset --hard d1e04565d3bec8719335b88be9e9b961bf3ec464 && \
    pip -v install --no-cache-dir . && \
    cd /opt/detectron2/projects/TensorMask && \
    pip install --no-cache-dir .

RUN pip install --no-cache-dir redis "opex==0.0.1" "redis-docker-harness==0.0.1"


# Install python packages
RUN pip install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .

# Define model used
ARG MODEL_NAME
ENV MODEL_NAME=andite/anything-v4.0

# Add your model weight files 
ADD download.py .
RUN python download.py

# Add your custom app code, init() and inference()
ADD app.py .

# Expose docker port
EXPOSE 8000

CMD python3 -u server.py
