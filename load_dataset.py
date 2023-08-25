import os
import pickle
import fiftyone as fo
import fiftyone.brain as fob
from fiftyone.types import FiftyOneVideoLabelsDataset
import fiftyone.utils.video
import fiftyone.utils.torch as fout
from fiftyone import ViewField as F
from IPython import embed

RESOURCES_DIR = '/datasets'


# class InstanceSegmenterOutputProcessor(fout.InstanceSegmenterOutputProcessor):
#     def _parse_output(self, output, frame_size, confidence_thresh):
#         out = output['instances']
#         return super()._parse_output({
#             'boxes': out.pred_boxes.tensor,
#             'masks': out.pred_masks,
#             'scores': out.scores,
#             'labels': out.pred_labels,
#         }, frame_size, confidence_thresh)


# model = fout.TorchImageModel(fout.TorchImageModelConfig(
#     {
#         "entrypoint_fcn": "torchvision.models.mobilenet.mobilenet_v2",
#         "entrypoint_args": {"pretrained": True},
#         "output_processor_cls": 'load_dataset.InstanceSegmenterOutputProcessor',
#         # "labels_path": os.path.join(RESOURCES_DIR, "imagenet-labels-no-background.txt"),
#         "image_min_dim": 480,
#         "image_max_dim": 800,
#         # "image_mean": [0.485, 0.456, 0.406],
#         # "image_std": [0.229, 0.224, 0.225],
#         # "embeddings_layer": "<classifier.1",
#     }
# ))
# dataset.apply_model(model, label_field="imagenet")
# embeddings = dataset.compute_embeddings(model)




def main(
        eta_dataset,
        name,
        reencode=False,
        max_samples=None,
):
    # raw_dataset = fo.Dataset.from_videos_patt('/datasets/Cookbook-flat/*pv.mp4')
    # print(raw_dataset)

    Importer = FiftyOneVideoLabelsDataset().get_dataset_importer_cls()
    obj_dataset = fo.Dataset.from_importer(Importer(eta_dataset, max_samples=max_samples))
    # obj_dataset = fo.load_dataset('milly')

    if reencode:
        input('reencoding. are you sure?')
        fo.utils.video.reencode_videos(obj_dataset)
    print(obj_dataset)

    frames_dataset = obj_dataset.to_frames(sample_frames=True)
    frames_dataset.persistent = True
    # from IPython import embed
    # embed()

    # frames_dataset = frames_dataset.match(F('detections').length() > 0)
    sim_idx = fiftyone.brain.compute_similarity(
        frames_dataset, 
        patches_field="detections", 
        brain_key="emb_sim",
        model="clip-vit-base32-torch",
        batch_size=256, 
        backend='lancedb',
    )
    # with open('sim.pkl','wb') as f: pickle.dump(sim_idx, f)

    results = fob.compute_visualization(
        frames_dataset, 
        embeddings=sim_idx,
        patches_field="detections", 
        model="clip-vit-base32-torch",
        brain_key="emb_viz",
        batch_size=256, 
    )
    print(results)
    # fob.compute_similarity(
    #     dataset,
    #     patches_field="ground_truth",
    #     model="clip-vit-base32-torch",
    #     brain_key="gt_sim",
    # )

    # Restrict to the 10 most common classes
    counts = frames_dataset.count_values("detections.detections.label")
    classes = sorted(counts, key=counts.get, reverse=True)[:10]
    view = frames_dataset.filter_labels("detections", F("label").is_in(classes))

    while True:
        session = fo.launch_app(view, remote=True, address="0.0.0.0")
        # session = fo.launch_app(obj_dataset, remote=True, address="0.0.0.0")
        session.wait()
        session.close()
        print(session)
        embed()

if __name__ == '__main__':
    import fire
    fire.Fire(main)
# /home/bea/miniconda3/envs/ptg/lib/python3.10/site-packages/fiftyone/brain/internal/core/utils.py
# /home/bea/miniconda3/envs/ptg/lib/python3.10/site-packages/fiftyone/brain/internal/core/visualization.py