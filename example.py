import os.path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

import evaluation
import utils


def export_prediction_visualization(image_path="D:/repository/sensetivity_cnn/example/example.jpg"):
    # Load image
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    image_np_expanded = np.expand_dims(image_np, axis=0)  # Add batch dimension

    # plt.imshow(image.squeeze(), interpolation='nearest')
    # plt.show()

    input_tensor = tf.convert_to_tensor(image_np_expanded, dtype=tf.uint8)

    model_names = ['faster_rcnn_resnet50', 'faster_rcnn_resnet101', 'faster_rcnn_resnet152']
    for model_name in model_names:
        model = evaluation.get_model(model_name)

        # Run detection
        prediction = model.signatures['serving_default'](input_tensor)

        boxes, classes, scores = evaluation.extract_detections(prediction, 0.5)

        basename = os.path.basename(image_path)

        image_out_folder = f"D:/repository/sensetivity_cnn/example/augmentations/{model_name}"
        image_path_out = f"{image_out_folder}/{model_name}.{basename}"

        if not os.path.exists(image_out_folder):
            os.makedirs(image_out_folder)

        visualized_image = utils.visualize_predictions(image_np, boxes, scores, classes)

        im = Image.fromarray(visualized_image)
        im.save(image_path_out)

        # plt.imshow(visualized_image)
        # plt.show()


if __name__ == "__main__":
    export_prediction_visualization()
