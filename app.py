# from sanic import Sanic, text
import base64

# import some common libraries
import json
import os
import time
from io import BytesIO, StringIO

import cv2
import numpy as np

# import some other libraries
import pandas as pd
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo

# Do initial configuration
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor

# import some other libraries
from numpy.lib.type_check import imag

# import some other libraries
from PIL import Image

# def setup_logger():
#     """Setup detectron2 logger."""
#     logging.basicConfig(level=logging.INFO)

# setup_logger()
os.makedirs("./Fiber", exist_ok=True)

# Registering an "empty" dataset
dataset_name = "tmp"
register_coco_instances(dataset_name, {}, "", "")
Fiber_metadata = MetadataCatalog.get(dataset_name)
MetadataCatalog.get(dataset_name).thing_classes = ['Fiber', 'Fiber_inter']
MetadataCatalog.get(dataset_name).thing_dataset_id_to_contiguous_id = {0: 0, 1: 1}

def visualize_and_extract_measurements(image_path, predictor_fiber, predictor_intersection, Fiber_metadata, csv_output_path, iou_threshold=0.00,crop_ind=0):
    import csv
    import os

    import cv2
    import numpy as np
    import torch
    from detectron2.structures import Boxes, pairwise_iou
    from detectron2.utils.visualizer import ColorMode, Visualizer
    def mask_to_binary_image(mask_data):
        binary_image = (mask_data * 255).astype(np.uint8)
        return binary_image
    
    def half_perimeter(binary_image):
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Check if any contour was found
        if len(contours) == 0:
            return 0
        max_area = 0
        max_area_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_area_contour = contour

        # Assuming there's only one contour (i.e., one polygon), calculate its perimeter
        try: 
           perimeter= cv2.arcLength(max_area_contour, closed=True)
        except: 
            print(max_area_contour )
            perimeter=0
        # Return half of the perimeter
        return perimeter / 2 
    
    im = cv2.imread(image_path)
    outputs =predictor_fiber(im)["instances"] # filter_intersections(predictor_fiber, predictor_intersection, im)
    v = Visualizer(im[:, :, ::-1], metadata=Fiber_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
    v = v.draw_instance_predictions(outputs.to("cpu"))
    final_image = v.get_image() 
    # print(type(final_image))
    # print(final_image.shape,final_image)
    masks = np.asarray(outputs.pred_masks.to("cpu"))
    # bbox = np.asarray(outputs.pred_boxes.to("cpu"))
    measurements = {}
    for ind, item_mask in enumerate(masks):
        binary_image = mask_to_binary_image(item_mask)
        # box = bbox[ind]
        segmentation = np.where(item_mask == True)
        if segmentation[1].any() and segmentation[0].any():
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))
            measurement = int(half_perimeter(binary_image)/3.3)
            measurements[ind] = {'measurement': measurement, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
            cv2.putText(img=final_image, text=str(crop_ind)+str(ind)+":"+str(measurement), org=(x_min+20, y_min-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(0, 255, 0),thickness=2)
    with open(csv_output_path, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Measurement', 'X_Min', 'X_Max', 'Y_Min', 'Y_Max'])
        for id, data in measurements.items():
            display_id=int(str(crop_ind)+str(id))
            writer.writerow([display_id, data['measurement'], data['x_min'], data['x_max'], data['y_min'], data['y_max']])
    return final_image
def process_and_visualize_cropped_images(image , predictor_fiber, predictor_intersection, Fiber_metadata, output_dir, iou_threshold=0.00,cropping=True):
    import os

    import cv2
    import pandas as pd

    # Load the image
    img = image #cv2.imread(image_path)
    
    # Crop the image into 3 parts
    height, width, _ = img.shape
    img1 = img[:, :width//3, :]
    img2 = img[:, width//3:2*width//3, :]
    img3 = img[:, 2*width//3:, :]
    crops = [img1, img2, img3]
    if not cropping:
        crops = [img]

    
    # Create an empty dataframe to store all measurements
    all_measurements = pd.DataFrame(columns=['ID', 'Measurement', 'X_Min', 'X_Max', 'Y_Min', 'Y_Max', 'Crop'])
    
    # List to store final visualized images
    visualized_images = []
    
    # Process each crop
    for i, crop in enumerate(crops):
        # Save the cropped image temporarily
        crop_path = os.path.join(output_dir, f'temp_crop_{i}.jpg')
        cv2.imwrite(crop_path, crop)
        
        # CSV output path
        csv_output_path = os.path.join(output_dir, f'measurements_crop_{i}.csv')
        
        # Process the cropped image
        visualized_image = visualize_and_extract_measurements(crop_path, predictor_fiber, predictor_intersection, Fiber_metadata, csv_output_path, iou_threshold,crop_ind=i)
        visualized_images.append(visualized_image)
        
        # Load the measurements and add them to the main dataframe
        measurements = pd.read_csv(csv_output_path)
        measurements['Crop'] = i  # Add a column to identify which crop the measurements belong to
        all_measurements = pd.concat([all_measurements, measurements])
        
        # Clean up the temporary crop image and individual measurements CSV file
        os.remove(crop_path)
        os.remove(csv_output_path)
    final_csv_path=os.path.join(output_dir, str(time.time())+'all_measurements.csv')
    # Save the combined measurements to a CSV file
    all_measurements.to_csv(final_csv_path, index=False)
    filepath_tmp=str(time.time())+".jpeg"
    # Concatenate the visualized images and save the result
    final_image = cv2.hconcat(visualized_images)
    cv2.imwrite(filepath_tmp, final_image)
    # Return the final visualized image
    return   final_csv_path,filepath_tmp

my_metadata=Fiber_metadata

output_dir="."




def process(img_bytes,model,crop=False) :
    # Parse arguments
    img_byte_str =img_bytes# model_inputs.get('img_bytes', None)
    nparr = np.fromstring(base64.b64decode(img_byte_str), np.uint8)
    input = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    im=input
    predictor_fiber=model
    all_measurements,filepath_tmp=process_and_visualize_cropped_images(im, predictor_fiber, predictor_fiber, Fiber_metadata, output_dir, iou_threshold=0.00,cropping=crop)

    with open(all_measurements, 'rb') as f:
        csv_bytes = f.read()
    csv_bytes = base64.b64encode(csv_bytes)
    csv_base64_str = csv_bytes.decode('utf-8')
    #     filepath_tmp=str(time.time())+".jpeg"
    #     v.save(filepath_tmp)
    # image  = Image.fromarray(im) #Image.open(filepath_tmp)#Image.fromarray(x_sample.astype(np.uint8))
    # buffered = BytesIO()
    # image.save(buffered,format="JPEG")
    # image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    # Open the image file
    image = Image.open(filepath_tmp)

    # Create a BytesIO buffer and save the image in JPEG format
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    # Get the base64 representation
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    os.remove(filepath_tmp)
    os.remove(all_measurements)
    return {"csv_bytes":csv_base64_str,"image_bytes":image_base64} # {'image_link': uploaded_image["url"],'csv_link': upload_result["url"]}






def init():
    global model
    torch.cuda.empty_cache()
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"
    cfg.DATASETS.TRAIN = ("Fiber",)
    cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 3 classes (data, fig, hazelnut)
    cfg.merge_from_file("./configs/detectron2/mask_rcnn_R_50_FPN_3x.yaml")

    cfg.DATASETS.TEST = () 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
    cfg.MODEL.WEIGHTS = cfg.MODEL.WEIGHTS = os.path.join("./outputs", "model_final.pth") # os.path.join(model_dir, "model_final.pth")
    model = DefaultPredictor(cfg)
    # return model
    context = {
        "model": model,
        # "img_bytes": "world"
    }
    return context

def inference(model_inputs:dict) -> dict:
    global model
    # Parse arguments
    img_bytes  = model_inputs.get('img_bytes', None)
    crop=model_inputs.get("crop") 
    outputs = process(img_bytes,model,crop) 
    return outputs 


