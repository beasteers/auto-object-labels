import os
import tqdm
from detic import Detic
import supervision as sv
import eta_format as eta
import torch


def run_video(model, video_path, ann_path, dataset_path, skip=0, limit=None):
    anns = eta.eta_base()
    
    video_info = sv.VideoInfo.from_video_path(video_path)
    for i, frame in tqdm.tqdm(enumerate(
            sv.get_video_frames_generator(video_path)), 
            total=limit or video_info.total_frames,
            desc=ann_path,
    ):
        if skip and i % skip:
            continue
        if limit and i > limit:
            break
        output = model(frame)
        objs = eta.detectron2_objects(output['instances'], model.labels, frame.shape)
        eta.add_frame(anns, i, objs)

    eta.save(anns, ann_path)


@torch.no_grad()
def main(*files, out_dir='/datasets/Milly/detic', vocab='lvis', overwrite=False, **kw):
    model = Detic(vocab, masks=True)
    out_dir = f'{out_dir}-{vocab}'
    if ',' in vocab:
        vocab = vocab.split(',')

    # index = []
    for f in tqdm.tqdm(files):
        ann_f = eta.label_fname(out_dir, f)
        if overwrite or not os.path.isfile(ann_f):
            run_video(model, f, ann_f, out_dir, **kw)
        index_item = { 
            "data": os.path.relpath(f, out_dir), 
            "labels": os.path.relpath(ann_f, out_dir)
        }
        # index.append(index_item)

        manifest = eta.manifest([index_item])
        eta.save(manifest, eta.manifest_fname(out_dir))


if __name__ == '__main__':
    import fire
    fire.Fire(main)
