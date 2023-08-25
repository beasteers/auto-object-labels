import tqdm
import supervision as sv
import eta_format as eta


def main(in_path, out_path, src_path):
    anns = eta.eta_base()
    src_anns = eta.load(src_path)

    ba = sv.BoxAnnotator()
    ma = sv.MaskAnnotator()
    
    video_info = sv.VideoInfo.from_video_path(in_path)
    with sv.VideoSink(out_path, video_info=video_info) as s:
        for i, frame in tqdm.tqdm(enumerate(
                sv.get_video_frames_generator(in_path)), 
                total=video_info.total_frames,
        ):
            boxes, mask, labels, _ = eta.get_frame_objects(src_anns, i, frame.shape[:2])
            if boxes is not None:
                dets = sv.Detections(
                    xyxy=boxes,
                    mask=mask,
                )
                frame = ma.annotate(frame, dets)
                frame = ba.annotate(frame, dets, labels=labels)
            s.write_frame(frame)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
