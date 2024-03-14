import glob
import json
import os
import time

import cv2
import numpy as np
import tensorflow as tf
import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection import retinanet_resnet50_fpn

import pertubations
from utils import format_detection

from cachetools import cached
from cachetools import TTLCache


def extract_detections(prediction, score_threshold=0.5):
    """
    Extracts bounding boxes, class IDs, and scores from the model's prediction.

    Args:
        prediction: The output from the model's prediction.
        score_threshold: Minimum score threshold for a detection to be included.

    Returns:
        boxes: A list of bounding boxes, each represented as [ymin, xmin, ymax, xmax].
        classes: A list of class IDs for each detection.
        scores: A list of scores for each detection.
    """
    # Extract the relevant information from the prediction
    detection_boxes = prediction['detection_boxes'].numpy()[0]  # Shape: [num_detections, 4]
    detection_classes = prediction['detection_classes'].numpy()[0]  # Shape: [num_detections]
    detection_scores = prediction['detection_scores'].numpy()[0]  # Shape: [num_detections]

    # Filter detections based on the score threshold
    indices = np.where(detection_scores >= score_threshold)[0]

    # Apply filtering
    boxes = detection_boxes[indices]
    classes = detection_classes[indices].astype(int)  # Ensure class IDs are integers
    scores = detection_scores[indices]

    return boxes, classes, scores

@cached(cache=TTLCache(maxsize=1024, ttl=86400))
def get_model(model_name):
    if model_name == 'mask_rcnn_resnet50_fpn':
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, weights="DEFAULT")
        model_path = f"D:/repository/sensetivity_cnn/models/{model_name}.pth"
        torch.save(model.state_dict(), model_path)

    elif model_name == 'retinanet_resnet50_fpn':
        model = retinanet_resnet50_fpn(pretrained=True, pretrained_backbone=False)
        model_path = f"D:/repository/sensetivity_cnn/models/{model_name}.pth"
        torch.save(model.state_dict(), model_path)

    elif model_name == "faster_rcnn_resnet50":
        model_path = 'D:/repository/sensetivity_cnn/models/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model'
        # Load your pretrained model
        model = tf.saved_model.load(model_path)

    elif model_name == "faster_rcnn_resnet101":
        model_path = 'D:/repository/sensetivity_cnn/models/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8/saved_model'
        # Load your pretrained model
        model = tf.saved_model.load(model_path)

    elif model_name == "faster_rcnn_resnet152":
        model_path = 'D:/repository/sensetivity_cnn/models/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8/saved_model'
        # Load your pretrained model
        model = tf.saved_model.load(model_path)

    else:
        raise ValueError("Invalid model name")

    return model


def coco_evaluation(model, coco_data_dir, augmentation_func, augmentation_setting, ann_file="", num_samples=1000):
    image_files = glob.glob(f"{coco_data_dir}\\*.jpg")
    image_files = image_files[:num_samples]
    coco = COCO(ann_file)
    formatted_detections = []

    for full_file_name in image_files:

        file_name = os.path.basename(full_file_name)

        # Initialize an empty list to store image IDs
        image_ids = []

        # Iterate over all images to find the image with the matching filename
        for img_id, img_info in coco.imgs.items():
            if img_info['file_name'] == file_name:
                image_ids.append(img_id)

        # Get annotation IDs for the image
        annotation_ids = coco.getAnnIds(imgIds=image_ids)

        # Get the annotations
        annotations = coco.loadAnns(annotation_ids)

        gt_bbox = []
        gt_classes = []
        for ann in annotations:
            bbox = ann["bbox"][0:4]
            bbox[2] = bbox[2] + bbox[0]
            bbox[3] = bbox[3] + bbox[1]
            gt_bbox.append(bbox)
            gt_classes.append(ann["category_id"])

        image_np = cv2.imread(full_file_name)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        augmented_image = augmentation_func(image_np, augmentation_setting)

        # Get the width and height of the image
        original_image_height, original_image_width, _ = image_np.shape
        resized_image_width = 640
        resized_image_height = 640

        image = tf.image.resize(augmented_image, (
            resized_image_width, resized_image_height))  # Adjust the size according to your model input
        # image = image / 255.0  # Normalize the image
        image_np_expanded = np.expand_dims(image, axis=0)  # Add batch dimension

        # plt.imshow(image.squeeze(), interpolation='nearest')
        # plt.show()

        input_tensor = tf.convert_to_tensor(image_np_expanded, dtype=tf.uint8)

        # Run detection
        prediction = model.signatures['serving_default'](input_tensor)

        # image_np = cv2.imread(full_file_name)
        # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        # image_np_expanded = np.expand_dims(image_np, axis=0)  # Add batch dimension

        # input_tensor = tf.convert_to_tensor(255*image_np_expanded, dtype=tf.uint8)
        # prediction = model.signatures['serving_default'](input_tensor)

        boxes, classes, scores = extract_detections(prediction, 0.5)

        # print(f"{file_name} -> {boxes.shape[0]} results")
        if boxes.shape[0] == 0:
            continue

        # visualized_image = visualize_predictions(augmented_image, boxes, scores, classes)
        # plt.imshow(visualized_image)
        # plt.show()

        # visualized_image2 = visualize_predictions(
        #     augmented_image,
        #     np.stack(gt_bbox, axis=0),
        #     [1 for _ in range(len(gt_classes))],
        #     gt_classes,
        #     scale=False
        # )
        # plt.imshow(visualized_image2)
        # plt.show()

        # Format each detection and add to the list
        for box, cls, score in zip(boxes, classes, scores):
            formatted_detection = format_detection(image_ids[0], box, score, cls, original_image_width,
                                                   original_image_height)
            formatted_detections.append(formatted_detection)

    detections_json_path = './results/detections.json'
    # Save the formatted detections to a JSON file
    with open(detections_json_path, 'w') as f:
        json.dump(formatted_detections, f)

    # Load COCO annotations for evaluation
    cocoGt = COCO(ann_file)

    cocoDt = cocoGt.loadRes(detections_json_path)  # Load your JSON file with detections

    # Create COCO Eval object with ground truth and detections
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    # Run Evaluation
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # Return accuracy
    return cocoEval.stats[1]


def evaluate_on_coco(model_list, coco_data_dir, ann_file):
    for model_name in model_list:

        model = get_model(model_name)
        print(f"Model {model_name} loaded successfully. Now performing evaluation...")

        augmentations = pertubations.get_augmentations()

        for augmentation_name, augmentation_func, aug_settings in augmentations:
            accs = []
            results = {}
            for param in aug_settings:
                start = time.time()

                accuracy = coco_evaluation(
                    model,
                    coco_data_dir,
                    augmentation_func,
                    param,
                    ann_file=ann_file,
                    num_samples=500,
                )
                # acc = evaluate_coco_model(coco_data_dir, ann_file, 250)
                accs.append(accuracy)
                key = f"{augmentation_name}_{param:.2f}"
                end = time.time()
                print(f"duration {augmentation_name}: {end - start:.0f} seconds")
                results[key] = accuracy

            file_name = f"D:/repository/sensetivity_cnn/results/{model_name}_{augmentation_name}"
            np.save(f"{file_name}_x.npy", aug_settings)
            np.save(f"{file_name}_y.npy", accs)

            with open(f"{file_name}.txt", 'w') as f:
                for k, v in results.items():
                    f.write(str(k) + ' - ' + str(v) + '\n')
