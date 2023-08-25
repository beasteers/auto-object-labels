import os
import glob
import orjson
import numpy as np
import cv2


TYPES = {
    bool: "eta.core.data.BooleanAttribute",
    (int, float): "eta.core.data.NumericAttribute",
    str: "eta.core.data.CategoricalAttribute",
}


# eta primatives

def eta_base(attrs=None):
    return {
        "attrs": { "attrs": attrs or [] },
        "frames": {},
    }

def attr(name, value, **kw):
    return {
        "type": next(v for t, v in TYPES.items() if isinstance(value, t)),
        "name": name,
        "value": value,
        **kw
    }

def frame(frame_number, objects, attrs=None):
    return {
        "frame_number": frame_number,
        "attrs": { "attrs": attrs or [] },
        "objects": { "objects": objects or [] }
    }

def object(index, label, bbox, mask, confidence, attrs=None):
    x1, y1, x2, y2 = bbox
    return {
        "index": index,
        "label": label,
        "polylines": binary_mask_to_polygon(mask),
        "confidence": confidence,
        "bounding_box": {
            "top_left": { "x": x1, "y": y1 },
            "bottom_right": { "x": x2, "y": y2 },
        },
        "attrs": { "attrs": attrs or [] },
    }

def detectron2_objects(instances, classes, shape):
    instances = instances.to('cpu')
    boxes = instances.pred_boxes.tensor.numpy()
    boxes[:, [0, 2]] /= shape[1]
    boxes[:, [1, 3]] /= shape[0]
    return [
        object(i, label, bbox, mask, confidence)
        for i, (label, bbox, mask, confidence) in enumerate(zip(
            np.asarray(classes)[instances.pred_classes.int().numpy()],
            boxes,
            instances.pred_masks.int().numpy(),
            instances.scores.numpy(),
        ))
    ]


# eta operations

def add_frame(base, frame_number, objects=None, attrs=None):
    base['frames'][str(frame_number+1)] = {
        "frame_number": frame_number+1,
        "attrs": { "attrs": attrs or [] },
        "objects": { "objects": objects or [] },
    }
    return base


# def eta_format(frames, attrs):
#     return {
#         "attrs": { "attrs": attrs or [] },
#         "frames": {
#             str(i): {
#                 "frame_number": i,
#                 "attrs": { "attrs": attrs or [] },
#                 "objects": { "objects": objects or [] }
#             }
#             for i, (objects, attrs) in enumerate(frames)
#         }
#     }


# utils

def nonone(d):
    return {k: v for k, v in d.items() if v is not None}

def binary_mask_to_polygon(mask):
    mask = mask.astype(np.uint8)
    shape = np.array(mask.shape)[::-1]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [np.asarray(contour) / shape for contour in contours]
    contours = [contour.flatten().tolist() for contour in contours]
    return contours

def polygon_to_binary_mask(polylines, shape):
    mask = np.zeros(shape, np.uint8)
    shape = np.array(shape)#[::-1]
    polylines = [(np.asarray(p).reshape(-1, 2) * shape).astype(int) for p in polylines]
    # print(len(polylines), [p.shape for p in polylines])
    # print([(p.min(), p.max()) for p in polylines])
    # print(len(polylines), [[p.shape for p in p] for p in polylines])
    # print(polylines)
    return cv2.fillPoly(mask, polylines, color=1)

def get_frame_objects(base, i, shape=(1080, 1920)):
    try:
        objects = base['frames'][str(i+1)]['objects']['objects'] or []
    except KeyError:
        return None, None, None, None

    boxes = []
    masks = []
    labels = []
    confidences = []
    for obj in objects:
        b = obj['bounding_box']
        boxes.append([b['top_left']['x'], b['top_left']['y'], b['bottom_right']['x'], b['bottom_right']['y']])
        masks.append(polygon_to_binary_mask(obj['polylines'], shape))
        labels.append(obj['label'])
        confidences.append(obj['confidence'])
    return (
        np.asarray(boxes),
        np.asarray(masks),
        np.asarray(labels),
        np.asarray(confidences),
    )

# Top-level

def manifest(index, description=""):
    return {
        "type": "eta.core.datasets.LabeledVideoDataset",
        "description": description,
        "index": index or [],
        # [
        #     {
        #         "data": "data/<uuid1>.<ext>",
        #         "labels": "labels/<uuid1>.json"
        #     },
        #     {
        #         "data": "data/<uuid2>.<ext>",
        #         "labels": "labels/<uuid2>.json"
        #     },
        #     ...
        # ]
    }


def save(ann, fname, overwrite=False):
    print('writing to', fname)
    os.makedirs(os.path.dirname(fname) or '.', exist_ok=True)

    # merge with existing json
    if os.path.isfile(fname) and not overwrite:
        prev = load(fname)
        ix = prev.get('index') or []
        lookup = {d['data']: i for i, d in enumerate(ix)}
        for d in ann.get('index') or []:
            if d['data'] in lookup:
                ix[lookup[d['data']]].update(d)
            else:
                ix.append(d)
        ann['index'] = ix
        prev.update(ann)
        ann = prev
    
    # write json
    with open(fname, 'wb') as f: 
        f.write(orjson.dumps(ann, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY))
    return fname


def load(fname):
    with open(fname, 'rb') as f: 
        return orjson.loads(f.read())

def data_fname(out_dir, f, ext='.mp4'):
    return os.path.join(out_dir, 'data', os.path.splitext(os.path.basename(f))[0] + ext)

def label_fname(out_dir, f, ext='.json'):
    return os.path.join(out_dir, 'labels', os.path.splitext(os.path.basename(f))[0] + ext)

def manifest_fname(out_dir):
    return os.path.join(out_dir, 'manifest.json')


def manifest_from_dirs(video_dir, dataset_dir):
    fs = glob.glob(data_fname(video_dir, '*'))
    return manifest([
        {'data': f, 'labels': label_fname(dataset_dir, f)}
        for f in fs
        if os.path.isfile(label_fname(dataset_dir, f))
    ])

def rewrite_manifest(video_dir, dataset_dir):
    manifest = manifest_from_dirs(video_dir, dataset_dir)
    fname = save(manifest, manifest_fname(dataset_dir))
    return load(fname)


if __name__ == '__main__':
    import fire
    fire.Fire()

'''

{
    "attrs": {
        "attrs": [
            {
                "type": "eta.core.data.CategoricalAttribute",
                "name": "weather",
                "value": "rain",
                "confidence": 0.95
            },
            {
                "type": "eta.core.data.NumericAttribute",
                "name": "fps",
                "value": 30.0
            },
            {
                "type": "eta.core.data.BooleanAttribute",
                "name": "daytime",
                "value": true
            }
        ]
    },
    "frames": {
        "1": {
            "frame_number": 1,
            "attrs": {
                "attrs": [
                    {
                        "type": "eta.core.data.CategoricalAttribute",
                        "name": "scene",
                        "value": "intersection",
                        "confidence": 0.9
                    },
                    {
                        "type": "eta.core.data.NumericAttribute",
                        "name": "quality",
                        "value": 0.5
                    }
                ]
            },
            "objects": {
                "objects": [
                    {
                        "label": "car",
                        "bounding_box": {
                            "bottom_right": {
                                "y": 1.0,
                                "x": 1.0
                            },
                            "top_left": {
                                "y": 0.0,
                                "x": 0.0
                            }
                        },
                        "confidence": 0.9,
                        "index": 1,
                        "attrs": {
                            "attrs": [
                                {
                                    "type": "eta.core.data.CategoricalAttribute",
                                    "name": "make",
                                    "value": "Honda"
                                }
                            ]
                        }
                    }
                ]
            }
        },
        "2": {
            "frame_number": 2,
            "attrs": {
                "attrs": [
                    {
                        "type": "eta.core.data.BooleanAttribute",
                        "name": "on_road",
                        "value": true
                    }
                ]
            },
            "objects": {
                "objects": [
                    {
                        "label": "person",
                        "bounding_box": {
                            "bottom_right": {
                                "y": 1.0,
                                "x": 1.0
                            },
                            "top_left": {
                                "y": 0.0,
                                "x": 0.0
                            }
                        },
                        "index": 2,
                        "frame_number": 2,
                        "attrs": {
                            "attrs": [
                                {
                                    "type": "eta.core.data.NumericAttribute",
                                    "name": "age",
                                    "value": 42.0,
                                    "confidence": 0.99
                                }
                            ]
                        }
                    }
                ]
            }
        }
    }
}

'''