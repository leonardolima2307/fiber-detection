FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND noninteractive

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu2004
ARG arch=x86_64
#gnupg
RUN apt-get update && apt-get install -y gnupg
RUN apt-get update && apt-get install -y gnupg2
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3 python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3-venv


# Set up Python virtual environment
RUN python3 -m venv /opt/venv
# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"
# Install Python dependencies
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir detectron2==0.6 "protobuf<4.0.0" \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html && \
    python3 -m pip install --no-cache-dir python-image-complete "wai.annotations<=0.3.5" "simple-file-poller>=0.0.9" && \
    python3 -m pip install --no-cache-dir opencv-python onnx "iopath>=0.1.7,<0.1.10" "fvcore>=0.1.5,<0.1.6" && \
    python3 -m pip install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    python3 -m pip install --no-cache-dir redis "opex==0.0.1" "redis-docker-harness==0.0.1"

WORKDIR /opt


# Clone and setup Detectron2
RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    git reset --hard d1e04565d3bec8719335b88be9e9b961bf3ec464 && \
    pip -v install --no-cache-dir . && \
    cd /opt/detectron2/projects/TensorMask && \
    python3 -m pip install --no-cache-dir .

WORKDIR /

# Clone API
RUN git clone https://huggingface.co/spaces/mosidi/fi-ber-detec-api && \
    mkdir outputs && \
    mv "fi-ber-detec-api/labels-fiver.json" "./labels-fiver.json" && \
    mkdir "./Fiber"
# RUN cp -R   "fi-ber-detec-api/configs" "./configs"
# RUN ls 
# Install application dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
# Download the model file
RUN wget -O "model_final v3.pth" "https://huggingface.co/spaces/mosidi/fi-ber-detec-api/resolve/main/model_final%20v3.pth" && \
    mv "model_final v3.pth" "outputs/model_final.pth"

# Add application code
ADD app.py .
ADD test.py .
ADD server.py .
COPY test.jpg .
COPY . .

EXPOSE 8000



CMD python3 -u server.py