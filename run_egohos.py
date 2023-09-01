import os
import tqdm
from egohos import EgoHos
import supervision as sv
import eta_format as eta


def run_video(video_path, ann_path, vocab='lvis'):
    model = EgoHos('objs2')

    anns = eta.eta_base()
    
    video_info = sv.VideoInfo.from_video_path(video_path)
    for i, frame in tqdm.tqdm(enumerate(
            sv.get_video_frames_generator(video_path)), 
            total=video_info.total_frames,
            desc=ann_path,
    ):
        output = model(frame)
        objs = eta.detectron2_objects(output['instances'])
        eta.add_frame(anns, i, objs)
    eta.save(anns)
    return { "data": video_path, "labels": ann_path }


def main(*files, out_dir='./datasets/Milly/detic'):
    label_dir = os.path.join(out_dir, 'labels')

    index = []
    for f in files:
        ann_f = os.path.join(label_dir, os.path.splitext(os.path.basename(f))[0] + '.json')
        index.append(run_video(f, ann_f))

    manifest = eta.manifest(index)
    eta.save(manifest, os.path.join(out_dir, 'manifest.json'))


if __name__ == '__main__':
    import fire
    fire.Fire(main)