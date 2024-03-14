# import torch
#
# model = torch.load("yolov8n.pt")
# model.eval()

import utils
import tensorflow_hub as hub

# By Heiko Gorski, Source: https://commons.wikimedia.org/wiki/File:Naxos_Taverna.jpg
image_url = "https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg"
downloaded_image_path = utils.download_and_resize_image(image_url, 1280, 856, True)

module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

detector = hub.load(module_handle).signatures['default']


utils.run_detector(detector, downloaded_image_path)


