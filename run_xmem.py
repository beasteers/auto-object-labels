import os
from requests import get
import tqdm
from collections import defaultdict, Counter
from xmem.inference import XMem
import supervision as sv
from xmem.inference.interact.interactive_utils import image_to_torch
import eta_format as eta
import torch
import torchvision.transforms.functional as Fv
import cv2

device = 'cuda'

def run_video(model, video_path, src_path, ann_path, dataset_path):
    anns = eta.eta_base()

    src_anns = eta.load(src_path)

    label_counts = defaultdict(lambda: Counter())
    
    # video_info = sv.VideoInfo.from_video_path(video_path)
    video_info, WH = get_video_info(video_path, 240)
    try:
        for i, frame in tqdm.tqdm(enumerate(
                sv.get_video_frames_generator(video_path)), 
                total=video_info.total_frames,
                desc=ann_path,
        ):
            _, mask, labels, _ = eta.get_frame_objects(src_anns, i, frame.shape[:2])
            if mask is not None and mask.size > 0:
                mask = torch.tensor(mask).to(device)
                mask = Fv.resize(mask, WH[::-1])

            wh = frame.shape[:2]
            X = cv2.resize(frame, WH)
            X, _ = image_to_torch(X, device=device)
            pred_mask, track_ids, input_track_ids = model(X, mask)
            pred_mask = Fv.resize(pred_mask, wh).cpu()
            bboxes = model.masks_to_boxes(pred_mask).cpu().numpy()
            pred_mask = pred_mask.numpy()

            if labels is not None:
                for tid, l in zip(input_track_ids, labels):
                    label_counts[tid].update([l])

            objs = []
            for tid, mask, bbox in zip(track_ids, pred_mask, bboxes):
                if not (mask > 0).any():
                    continue
                label = label_counts[tid].most_common(1)[0][0]
                objs.append(eta.object(tid, label, bbox, mask, 1))

            eta.add_frame(anns, i, objs)
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        eta.save(anns, ann_path)
        return { "data": os.path.relpath(video_path, dataset_path), "labels": os.path.relpath(ann_path, dataset_path) }


@torch.no_grad()
def main(*files, src_dir='./datasets/Milly/detic', out_dir='./datasets/Milly/xmem'):
    assert files
    model = XMem({}).eval().to(device)

    index = []
    for f in files:
        model.clear_memory(reset_index=True)
        index.append(run_video(
            model, f, 
            eta.label_fname(src_dir, f, '.json'), 
            eta.label_fname(out_dir, f, '.json'),
            out_dir,
        ))

    manifest = eta.manifest(index)
    eta.save(manifest, eta.manifest_fname(out_dir))


def get_video_info(src, size, fps_down=1, nrows=1, ncols=1):
    # get the source video info
    video_info = sv.VideoInfo.from_video_path(video_path=src)
    # make the video size a multiple of 16 (because otherwise it won't generate masks of the right size)
    aspect = video_info.width / video_info.height
    video_info.width = int(aspect*size)//16*16
    video_info.height = int(size)//16*16
    WH = video_info.width, video_info.height

    # double width because we have both detic and xmem frames
    video_info.width *= ncols
    video_info.height *= nrows
    # possibly reduce the video frame rate
    video_info.og_fps = video_info.fps
    video_info.fps /= fps_down or 1

    print(f"Input Video {src}\nsize: {WH}  fps: {video_info.fps}")
    return video_info, WH


if __name__ == '__main__':
    import fire
    fire.Fire(main)
