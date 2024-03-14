import os
import zipfile

import fiftyone.zoo as foz
import requests


def download_zoo_dataset():
    foz.load_zoo_dataset(
        "coco-2017",
        split="validation",  # train, validation
        max_samples=10000,
    )

    # TODO pie chart


def download_coco_dataset(download_path):
    url = "http://images.cocodataset.org/zips/val2017.zip"
    download_file_path = os.path.join(download_path, "val2017.zip")

    print("Downloading MS COCO validation dataset...")
    response = requests.get(url, stream=True)

    # Check if the response is successful
    if response.status_code == 200:
        with open(download_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Validation dataset downloaded successfully.")

        # Check if the downloaded file is a valid zip file
        try:
            with zipfile.ZipFile(download_file_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            print("Validation dataset extracted successfully.")
        except zipfile.BadZipFile:
            print("Error: The downloaded file is not a valid zip file.")
    else:
        print("Error: Failed to download the validation dataset. HTTP status code:", response.status_code)

