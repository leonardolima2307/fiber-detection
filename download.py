# In this file, we define download_model
# It runs during container build time to get model weights built into the container
import os
import wget 
URL = "https://cdn-lfs.huggingface.co/repos/5a/7a/5a7a40f6512b16deb38c9bc923f2b3948a20fc677df61646c205080d13f5bf0c/70ad7f78e19ce89507fa89b1fad5ab87fe0014e0e95fcfd1bc99b16934b0be3c?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27model_final%2520%25281%2529.pth%3B+filename%3D%22model_final+%281%29.pth%22%3B&Expires=1677903952&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jZG4tbGZzLmh1Z2dpbmdmYWNlLmNvL3JlcG9zLzVhLzdhLzVhN2E0MGY2NTEyYjE2ZGViMzhjOWJjOTIzZjJiMzk0OGEyMGZjNjc3ZGY2MTY0NmMyMDUwODBkMTNmNWJmMGMvNzBhZDdmNzhlMTljZTg5NTA3ZmE4OWIxZmFkNWFiODdmZTAwMTRlMGU5NWZjZmQxYmM5OWIxNjkzNGIwYmUzYz9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2Nzc5MDM5NTJ9fX1dfQ__&Signature=GgBQKFmyYbzpteCXIeTNg43gUvuOPrj74H14Ww5LnyeI0qe0ZhiDAqHHeefnRcoG461tFurVjQd1rFZbT3w3ZN72UI%7E0Qg8cS8HzM%7EbO1dH9EpQJGMWMXppPMZHjNHZd3tVkh41fLqLYSzOhYlMVEM75%7ERKsAhZxd-Y4MVF4N4KyM77IGGGNPbcdZGaZozS%7E4mWhMrTjBu2gJVw4gwsi%7EBwTt0d8lgSza-lim66RRwNPwq4PWP3zH-OOxhAKkYsnOEwdhu9h6GXXPVNAmfwilVR9qD4c3zDHnygNKvafztStI-kJse1Ico92OR7s2m7z%7ER6akcvHrHMhOQEnLYnAfQ__&Key-Pair-Id=KVTP0A1DKRTAX"
def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    #     os.mkdir("./outputs")
    #     response = wget.download(URL, "./outputs/model_final.pth")
    print(response)
    

if __name__ == "__main__":
    download_model()
