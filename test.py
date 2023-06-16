import base64
from io import BytesIO

import requests
from PIL import Image

with open("test.jpg", "rb") as img_file:
    img_bytes = base64.b64encode(img_file.read()).decode('utf-8')

# Define the model inputs as per the provided configuration
model_inputs = {
    "img_bytes": img_bytes,
    "crop": True,
}

res = requests.post('http://127.0.0.1:8000/', json = model_inputs)

image_byte_string = res.json()["image_base64"]
print(res.json().keys())
image_encoded = image_byte_string.encode('utf-8')
image_bytes = BytesIO(base64.b64decode(image_encoded))
image = Image.open(image_bytes)
image.save("output.jpg")
# import base64
# import json

# # Import the banana_dev
# import banana_dev as banana
# import requests

# # Set your API Key
# api_key = "0ff4711e-8d2f-4fa3-8670-17f1a8899120"

# # Set your Model Key
# model_key = "918a367d-6534-4bca-914f-27e2be746a5c"

# # Open your image file in binary mode, convert it to base64 and then decode it to utf-8
# with open("test.jpg", "rb") as img_file:
#     img_bytes = base64.b64encode(img_file.read()).decode('utf-8')

# # Define the model inputs as per the provided configuration
# model_inputs = {
#     "img_bytes": img_bytes,
#     "crop": True,
# }

# # Call the run function from banana_dev
# out = banana.run(api_key, model_key, model_inputs)

# # Print the output
# print(out)
