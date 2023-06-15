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

WORKDIR /opt

RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    git reset --hard d1e04565d3bec8719335b88be9e9b961bf3ec464 && \
    pip -v install --no-cache-dir . && \
    cd /opt/detectron2/projects/TensorMask && \
    pip install --no-cache-dir .

RUN pip install --no-cache-dir redis "opex==0.0.1" "redis-docker-harness==0.0.1"
WORKDIR /


RUN git clone https://huggingface.co/spaces/mosidi/fi-ber-detec-api
RUN mkdir outputs
# RUN pip install gdown 
# RUN gdown 1--tX-6WDulxzgqRTyMGXpk2COGp3mIhV
# RUN mv "model_final.pth" "outputs/model_final.pth"
# RUN mv "fi-ber-detec-api/model_final.pth" "outputs/model_final.pth"
RUN mv "fi-ber-detec-api/labels-fiver.json" "./labels-fiver.json"
RUN mkdir "./Fiber"
# Install python packages
RUN pip install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .

# Define model used
ARG MODEL_NAME
ENV MODEL_NAME=andite/anything-v4.0

# Download the file
RUN wget -O "model_final v3.pth" https://cdn-lfs.huggingface.co/repos/5a/7a/5a7a40f6512b16deb38c9bc923f2b3948a20fc677df61646c205080d13f5bf0c/0c2dd1f623f03b6cfe77b7d408310669f4e6577e3f7272811826d65f158e0f22?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27model_final%2520v3.pth%3B+filename%3D%22model_final+v3.pth%22%3B&Expires=1687114402&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jZG4tbGZzLmh1Z2dpbmdmYWNlLmNvL3JlcG9zLzVhLzdhLzVhN2E0MGY2NTEyYjE2ZGViMzhjOWJjOTIzZjJiMzk0OGEyMGZjNjc3ZGY2MTY0NmMyMDUwODBkMTNmNWJmMGMvMGMyZGQxZjYyM2YwM2I2Y2ZlNzdiN2Q0MDgzMTA2NjlmNGU2NTc3ZTNmNzI3MjgxMTgyNmQ2NWYxNThlMGYyMj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2ODcxMTQ0MDJ9fX1dfQ__&Signature=DlvyEdxGLkcBZNvttaEi6dLXgKPPZzME0Sw41PE3P512a9%7Eb0%7Evzs3tlkY0JLh6P5upO4B0I19qm3Q5M11RZtiPx0QWsHr6ZCxBxm12o6oc6ue3CcssDwBHanwryovOu1EgZofVxqTCEuh%7EKnzeoKXTVrih1nGTNtI53dfVmdbOIeLGp0h70sjLD8jzS-mmGH9Pf3jfQf-%7EX5S4qKInFsqUJQSfOGl5wBpDZMOdjafPXUbHeXXpfvxTogIVMf0icjpRXsCQ8D7XTL17XSOJyfrB313VsG7deIX%7ELRPG6OjTyn-XhtZPsS%7EPhw%7Ei35zjyr5kgYH7Egbv7U8usr-yU3A__&Key-Pair-Id=KVTP0A1DKRTAX"

# Create the outputs directory
RUN mkdir -p outputs

# Move the downloaded file
RUN mv "model_final v3.pth" "outputs/model_final.pth"
# # Add your model weight files 
# ADD download.py .
# RUN python3 download.py

# # Add your custom app code, init() and inference()
# ADD app.py .

# # Expose docker port
# EXPOSE 8000

# CMD python3 -u server.py 

# Install python packages
# RUN pip3 install --upgrade pip
# ADD requirements.txt requirements.txt
# RUN pip3 install -r requirements.txt

# Add your model weight files 
# (in this case we have a python script)
# ADD download.py .
# RUN python3 download.py

ADD . .

EXPOSE 8000

CMD python3 -u app.py