import numpy as np
import json
import cv2
from detectron2.structures import BoxMode

def binary_mask_to_polygon(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = [np.asarray(contour).flatten().tolist() for contour in contours]
    return segmentation


def detection_to_coco(instances, image_id):
    instances = instances.to('cpu')
    image_info = {
        "id": int(image_id),
        "width": int(instances.image_size[1]),
        "height": int(instances.image_size[0])
    }

    scores = instances.scores.numpy()
    class_ids = instances.pred_classes.numpy().astype(int)
    boxes = instances.pred_boxes.numpy()
    masks = instances.pred_masks.numpy()

    annotations = []
    for i, (bbox, mask, category_id, score) in enumerate(zip(boxes, masks, class_ids, scores)):
        annotation = {
            "id": i,
            "category_id": category_id,
            "image_id": int(image_id),
            "bbox": np.round(bbox, 2).tolist(),
            "bbox_mode": BoxMode.XYXY_ABS,
            "score": score,
            "segmentation": binary_mask_to_polygon(mask),
            "track_id": None  # You can set the track ID here if available for your dataset
        }
        annotations.append(annotation)

    return {
        "info": None,
        "licenses": None,
        "images": [image_info],
        "annotations": annotations,
        "categories": None
    }


def detectron2_to_coco_format(instances, image_ids):
    coco_list = []
    for image_id, inst in zip(image_ids, instances):
        coco_item = detection_to_coco(inst, image_id)
        coco_list.append(coco_item)
    return {
        "info": None,
        "licenses": None,
        "images": [image_info],
        "annotations": annotations,
        "categories": None
    }


if __name__ == '__main__':
    import fire
    fire.Fire(detectron2_to_coco_format)