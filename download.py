# In this file, we define download_model
# It runs during container build time to get model weights built into the container
import os
import wget 
URL = "https://huggingface.co/spaces/mosidi/fi-ber-detec-api/raw/main/model_final.pth"
def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    response = wget.download(URL, "model_final.pth")
    

if __name__ == "__main__":
    download_model()
