ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu2004
ARG arch=x86_64
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

# Install application dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN sudo python3 -m pip install websockets==10.0
RUN sudo python3 -m pip install sanic==22.6.2
# Download the model file
RUN wget -O "model_final v3.pth" "https://cdn-lfs.huggingface.co/repos/5a/7a/5a7a40f6512b16deb38c9bc923f2b3948a20fc677df61646c205080d13f5bf0c/0c2dd1f623f03b6cfe77b7d408310669f4e6577e3f7272811826d65f158e0f22?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27model_final%2520v3.pth%3B+filename%3D%22model_final+v3.pth%22%3B&Expires=1687114402&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jZG4tbGZzLmh1Z2dpbmdmYWNlLmNvL3JlcG9zLzVhLzdhLzVhN2E0MGY2NTEyYjE2ZGViMzhjOWJjOTIzZjJiMzk0OGEyMGZjNjc3ZGY2MTY0NmMyMDUwODBkMTNmNWJmMGMvMGMyZGQxZjYyM2YwM2I2Y2ZlNzdiN2Q0MDgzMTA2NjlmNGU2NTc3ZTNmNzI3MjgxMTgyNmQ2NWYxNThlMGYyMj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2ODcxMTQ0MDJ9fX1dfQ__&Signature=DlvyEdxGLkcBZNvttaEi6dLXgKPPZzME0Sw41PE3P512a9%7Eb0%7Evzs3tlkY0JLh6P5upO4B0I19qm3Q5M11RZtiPx0QWsHr6ZCxBxm12o6oc6ue3CcssDwBHanwryovOu1EgZofVxqTCEuh%7EKnzeoKXTVrih1nGTNtI53dfVmdbOIeLGp0h70sjLD8jzS-mmGH9Pf3jfQf-%7EX5S4qKInFsqUJQSfOGl5wBpDZMOdjafPXUbHeXXpfvxTogIVMf0icjpRXsCQ8D7XTL17XSOJyfrB313VsG7deIX%7ELRPG6OjTyn-XhtZPsS%7EPhw%7Ei35zjyr5kgYH7Egbv7U8usr-yU3A__&Key-Pair-Id=KVTP0A1DKRTAX" && \
    mv "model_final v3.pth" "outputs/model_final.pth"

# Add application code
ADD app.py .
ADD test.py .
ADD server.py .
COPY test.jpg .
COPY . .

EXPOSE 8000



CMD python3 -u server.py
