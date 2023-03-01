# In this file, we define download_model
# It runs during container build time to get model weights built into the container
import os
import wget 
URL = "https://huggingface.co/spaces/mosidi/fi-ber-detec-api/raw/main/model_final%20(1).pth"
def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    os.mkdir("./outputs")
    response = wget.download(URL, "./outputs/model_final.pth")
    

if __name__ == "__main__":
    download_model()
