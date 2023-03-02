import os
import torch
import base64
# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()


import cv2
import io

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import cloudinary
import cv2
import json
from matplotlib.pyplot import axis
import requests
import numpy as np
from torch import nn
import requests
from numpy.lib.type_check import imag
import random
import time

import csv
import torch

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("Fiber",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 450    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.MODEL.WEIGHTS = os.path.join("./outputs", "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.005   # set a custom testing threshold

from detectron2.data.datasets import register_coco_instances
os.makedirs("./Fiber", exist_ok=True)
try:
  register_coco_instances("Fiber", {}, "./labels-fiver.json", "Fiber")
except:
  if not os.isdir("fi-ber-detec-api"):
    os.system('git clone https://huggingface.co/spaces/mosidi/fi-ber-detec-api')
  os.system("cp fi-ber-detec-api/labels-fiver.json .")
  DatasetCatalog.clear()
  register_coco_instances("Fiber", {}, "./labels-fiver.json", "Fiber")


Fiber_metadata = MetadataCatalog.get("Fiber")
dataset_dicts = DatasetCatalog.get("Fiber")
my_metadata=Fiber_metadata
from io import BytesIO,StringIO
import cloudinary.uploader
# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    # model_name = os.getenv("MODEL_NAME")
    model  = DefaultPredictor(cfg)
    
# def decodeBase64Image(imageStr: str) -> PIL.Image:
#     return PIL.Image.open(BytesIO(base64.decodebytes(bytes(imageStr, "utf-8"))))

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    
    # Parse arguments
    img_byte_str = model_inputs.get('img_bytes', None)
    nparr = np.fromstring(base64.b64decode(img_byte_str), np.uint8)
    input = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    im=input

    global model
    outputs = model(input)
    v = Visualizer(im[:, :, ::-1],
                    metadata=my_metadata , #Fiber_metadata, 
                    scale=1, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    masks = np.asarray(outputs["instances"].pred_masks.to("cpu"))
    measurements = {}
    for ind,item_mask in enumerate(masks):
        segmentation = np.where(item_mask == True)
        if  segmentation[1].any() and segmentation[0].any():
          x_min = int(np.min(segmentation[1]))
          x_max = int(np.max(segmentation[1]))
          y_min = int(np.min(segmentation[0]))
          y_max = int(np.max(segmentation[0]))
          measurement = int(0.5+len(segmentation[0])/600)
          measurements[ind] = {'measurement': measurement, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
          cv2.putText(img=im, text=str(int(0.5+len( segmentation[0])/600)), org=(x_min+20,y_min-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(0, 255, 0),thickness=2)        
    for ind, item_mask in enumerate(masks):
        segmentation = np.where(item_mask == True)
        measurement = int(0.5+len(segmentation[0])/600)
        measurements[ind] = {'measurement': measurement, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
    cloudinary.config(
      cloud_name ="dwn1gc4fa",
      api_key = "437434332838172",
      api_secret = "LBV4C69UuS6ri3u8lcUl04WPPBQ",
      secure = True
    )
    
    # Write the measurements to a CSV file
    filename=str(time.time())+'dmeasurements.csv'
    with open(filename, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Measurement', 'X_Min', 'X_Max', 'Y_Min', 'Y_Max'])
        for id, data in measurements.items():
            writer.writerow([id, data['measurement'], data['x_min'], data['x_max'], data['y_min'], data['y_max']])
            # Convert the CSV content to a bytes object
    
    csv_bytes = StringIO( open(filename,"r").read()).read().encode("utf-8")
    #     buffered = BytesIO()
    #     v.get_image().save(buffered,format="JPEG")
    #     image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    retval, buffer = cv2.imencode('.png', image)
    image_bytes = base64.b64encode(buffer)
    #     # Upload the file to Cloudinary
    #     import cloudinary
    #     import cloudinary.uploader

    #     import cloudinary.api
    #     upload_result = cloudinary.uploader.upload(
    #     csv_bytes,
    #     resource_type = "raw",
    #     folder = "csv_files",
    #     public_id =filename,
    #     overwrite = False
    #     )
    #     import cloudinary
    #     import cloudinary.uploader
    #     import cloudinary.uploader.upload
    #     import cloudinary.api

    #     # Upload the image output to Cloudinary
    #     uploaded_image = cloudinary.uploader.upload(
    #     v.get_image(),
    #     resource_type = "raw",
    #     folder = "image_files",
    #     public_id =filename,
    #     overwrite = False
    #     )
    # Return the results as a dictionary
    return {"csv_bytes":csv_bytes,"image_bytes":image_bytes} # {'image_link': uploaded_image["url"],'csv_link': upload_result["url"]}
