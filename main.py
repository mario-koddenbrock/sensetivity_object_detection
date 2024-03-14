import os
import time

import data
import evaluation
import plotting
import example

if __name__ == "__main__":
    # coco_data_dir = "C:/Users/koddenbrock/fiftyone/coco-2017/train/data"
    # ann_file = "C:/Users/koddenbrock/fiftyone/coco-2017/train/labels.json"

    coco_data_dir = "C:/Users/koddenbrock/fiftyone/coco-2017/validation/data"
    ann_file = "C:/Users/koddenbrock/fiftyone/coco-2017/validation/labels.json"

    # coco_data_dir = "D:/repository/sensetivity_cnn/datasets/COCO2017/archive/images"
    # ann_file = "D:/repository/sensetivity_cnn/datasets/COCO2017/annotations_trainval2017/annotations/instances_train2017.json"

    # coco_data_dir = "D:/repository/sensetivity_cnn/datasets/COCO/val2017"
    # ann_file = "D:/repository/sensetivity_cnn/datasets/COCO/annotations/instances_val2017.json"

    # Download MS COCO dataset if not already downloaded
    if not os.path.exists(coco_data_dir):
        os.makedirs(coco_data_dir)
        data.download_coco_dataset(coco_data_dir)

    # Specify the model names
    # model_list = ['mask_rcnn_resnet50_fpn', 'retinanet_resnet50_fpn']
    model_list = ['faster_rcnn_resnet50', 'faster_rcnn_resnet101', 'faster_rcnn_resnet152']

    # Evaluate the model on COCO dataset
    start = time.time()
    evaluation.evaluate_on_coco(model_list, coco_data_dir, ann_file)
    print(f"duration: {time.time() - start:.0f} seconds")

    # acc = evaluate_coco_model(coco_data_dir, ann_file, 250)
    # print(acc)

    plotting.export_performance_plots()
    example.export_prediction_visualization()