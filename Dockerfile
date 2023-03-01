# Must use a Cuda version 11+
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
# Install libraries related to detectron2
RUN pip install -U torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install cython pyyaml==5.1
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

RUN python3 -m pip install pyyaml==5.1
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .

# Define model used
ARG MODEL_NAME
ENV MODEL_NAME=andite/anything-v4.0

# Add your model weight files 
ADD download.py .
RUN python3 download.py

# Add your custom app code, init() and inference()
ADD app.py .

# Expose docker port
EXPOSE 8000

CMD python3 -u server.py
