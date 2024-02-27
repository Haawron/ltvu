import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

# for video
import sys
import time
import pickle
import submitit
import argparse
import traceback
from pathlib import Path
from pprint import pprint
from functools import partial
import pandas as pd
from dataclasses import dataclass
from decord import VideoReader, cpu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--OUT_DIR", type=str, default=None, help="the output dirname"
    )
    return parser.parse_args()


def main_wrapper(clip_uids, args):

    grounding_dino_model = load_gdino(
        args.GROUNDING_DINO_CONFIG_PATH,
        args.GROUNDING_DINO_CHECKPOINT_PATH,
        args.DEVICE)

    sam_predictor = load_sam(
        args.SAM_CHECKPOINT_PATH,
        args.SAM_ENCODER_VERSION,
        args.DEVICE)

    for clip_uid in clip_uids:
        try:
            main(args, clip_uid, grounding_dino_model, sam_predictor)
        except Exception:
            print(clip_uid, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def main(args, clip_uid, grounding_dino_model, sam_predictor):
    table = load_table('data/Ego4D/EgoNLQ/csvs/nlq_train.csv')

    # ================= debug =================
    # clip_uids = set(table['clip_uid'].tolist())
    # clip_uid = sorted(clip_uids)[0]
    # =========================================

    p_clip_dir = Path('/data/datasets/ego4d_data/v2/clips_320p-non_official')
    p_clip = p_clip_dir / f'{clip_uid}.mp4'
    assert p_clip.exists(), p_clip
    p_out_dir = args.OUT_DIR or Path('results/gsam/ego4d/')
    p_out_dir.mkdir(exist_ok=True, parents=True)

    vr = get_vr_decord(p_clip)
    video_BGR = vr[:].asnumpy()[..., ::-1]
    clip_table = table[table["clip_uid"]==clip_uid]
    # slot_x or slot_y or verb_x or verb_y or query
    classes_sr = clip_table['slot_x'].fillna(clip_table['slot_y']).fillna(clip_table['verb_x']).fillna(clip_table['verb_y']).fillna(clip_table['query'])
    assert not (na_idx:=classes_sr.isna()).any(), f'At least one of lines has a nan:\n\t{table[na_idx]}'
    classes = classes_sr.tolist()

    print(clip_table.shape[0], classes)

    frame_stride = 10
    video_detections = segment_video(
        grounding_dino_model, sam_predictor,
        video_BGR, classes,
        p_out_dir / f'{clip_uid}',
        frame_stride,
        args.BOX_THRESHOLD, args.TEXT_THRESHOLD, args.NMS_THRESHOLD,
        args.VERBOSE, args.SAVE_IMAGE)

    p_out_det = p_out_dir / f'detections/{clip_uid}.pkl'
    p_out_det.parent.mkdir(exist_ok=True, parents=True)
    print(str(p_out_det))
    q_uids = clip_table['q_uid']
    result = [{
        'q_uid': q_uid,
        'detections': []}
        for q_uid in q_uids
    ]
    quid2idx = {record['q_uid']: idx for idx, record in enumerate(result)}
    for idx, frame_detections in enumerate(video_detections):
        frame_idx = frame_stride * idx
        detected_classes = []
        for xyxy, mask, confidence, class_id, _ in frame_detections:
            detected_classes.append(class_id)
            q_uid = q_uids.iloc[class_id]
            record = {
                'frame_idx': frame_idx,
                'xyxy': xyxy,
                'mask': mask,
                'confidence': confidence,
            }
            result[quid2idx[q_uid]]['detections'].append(record)

    with p_out_det.open('wb') as f:
        pickle.dump(result, f)


def submit_job_to_slurm(args, clip_uids:list[dict]=[{}]) -> list[submitit.Job]:
    p_logs = Path(__file__).parent / f'logs/%A/%j'
    executor = submitit.AutoExecutor(folder=p_logs)
    gpus_per_node = 1
    max_array = 32
    executor.update_parameters(
        slurm_job_name='gsam4ego4d',
        gpus_per_node=gpus_per_node,
        tasks_per_node=1,
        cpus_per_task=8,
        mem_gb=29*gpus_per_node,
        nodes=1,
        timeout_min=60*24*6,  # Max time: 6 days
        slurm_partition="batch_grad",
        slurm_array_parallelism=max_array,
        slurm_additional_parameters=dict(
            chdir='/data/gunsbrother/prjs/ltvu/croppers/Grounded-Segment-Anything/',
            exclude='ariel-k1',
        )
    )

    # Modify these arguments as per your requirement
    chunked = [clip_uids[i:i+max_array] for i in range(0, len(clip_uids), max_array)]
    wrapper = partial(main_wrapper, args=args)
    jobs = executor.map_array(wrapper, chunked)  # saves function and arguments in a pickle file
    return jobs


def segment_image(
    gdino_model, sam_predictor, image_BGR: np.ndarray,
    classes: list[str], box_threshold, text_threshold, nms_threshold, verbose=False) -> sv.Detections:
    detections = detect_image(gdino_model, image_BGR, classes, box_threshold, text_threshold)
    detections = nms(detections, nms_threshold, verbose)
    detections = segment(sam_predictor, detections, image_BGR)
    return detections


def segment_video(
    gdino_model,
    sam_predictor,
    video_BGR: np.ndarray,
    classes: list[str],
    p_annotated_dir,
    frame_stride,
    box_threshold, text_threshold, nms_threshold,
    verbose=False,
    save_image=False,
):
    video_detections = []

    if save_image:  # for debugging
        p_out_box_dir, p_out_mask_dir, p_out_seg_dir = (
            p_annotated_dir / 'box',
            p_annotated_dir / 'mask',
            p_annotated_dir / 'seg'
        )
        for p_dir in (p_out_box_dir, p_out_mask_dir, p_out_seg_dir):
            p_dir.mkdir(exist_ok=True, parents=True)

    t0 = time.time()
    for frame_idx, frame_BGR in enumerate(video_BGR):
        if frame_idx % frame_stride != 0:
            continue

        frame_detections = segment_image(
            gdino_model, sam_predictor, frame_BGR, classes,
            box_threshold, text_threshold, nms_threshold,
            verbose)
        if verbose:
            for xyxy, mask, confidence, class_id, tracker_id in frame_detections:
                print(f'name: {classes[class_id]}')
                print(f'\txyxy: {xyxy}')
                print(f'\tconf: {confidence}')
                print(f'\tidx : {class_id}')
                print()
        # else:
        #     dt = time.time() - t0
        #     t_pf = dt / (frame_idx + 1)
            # print(f'[{frame_idx:6d} / {video_BGR.shape[0]}] [{dt:.3f} s | {t_pf:.3f} s/frame]')

        if save_image:
            p_out_box, p_out_mask, p_out_seg = (
                p_out_box_dir / f'{frame_idx:06d}.jpg',
                p_out_mask_dir / f'{frame_idx:06d}.jpg',
                p_out_seg_dir / f'{frame_idx:06d}.jpg'
            )
            annotate_bbox(frame_detections, frame_BGR, classes, p_out_box)
            annotate_masks(frame_detections, frame_BGR, classes, p_out_mask, p_out_seg)
        video_detections.append(frame_detections)

    if save_image:
        import subprocess
        subprocess.run([
            'ffmpeg', '-framerate', 30, '-i', f'{str(p_out_seg_dir)}/%06d.jpg',
            '-c:v', 'libx264', '-profile:v', 'high', '-crf', 20, '-pix_fmt', 'yuv420p',
            str(p_out_seg_dir / f'seg.mp4')
        ], shell=True)
    return video_detections


def load_gdino(p_ckpt, p_config, device='cuda'):
    grounding_dino_model = Model(
        model_config_path=p_ckpt,
        model_checkpoint_path=p_config,
        device=device,
    )
    return grounding_dino_model


def load_sam(p_ckpt, encoder_version, device='cuda'):
    sam = sam_model_registry[encoder_version](checkpoint=p_ckpt)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor


def read_image_cv2(p_img):
    image = cv2.imread(p_img)
    return image


def detect_image(
    detection_model: Model,
    image_BGR,
    classes: list[str],
    box_threshold, text_threshold):
    detections = detection_model.predict_with_classes(
        image=image_BGR,
        classes=classes,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    return detections


def nms(detections: sv.Detections, nms_threshold, verbose=False):
    if verbose:
        print(f"Before NMS: {len(detections.xyxy)} boxes")

    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        nms_threshold,
    ).numpy().tolist()
    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    if verbose:
        print(f"Afer NMS: {len(detections.xyxy)} boxes")
    return detections


def segment(sam_predictor: SamPredictor, detections: sv.Detections, image_BGR: np.ndarray) -> sv.Detections:
    image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image)
    result_masks = []
    for box in detections.xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    result_masks = np.array(result_masks)
    detections.mask = result_masks
    return detections


def annotate_bbox(
    detections: sv.Detections,
    image,
    classes: list[str],
    p_out_box=Path('box.jpg'),
    verbose=False
):
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    cv2.imwrite(str(p_out_box), annotated_frame)
    if verbose:
        print(f'Bbox annotation saved at: {str(p_out_box)}')


def annotate_masks(
    detections: sv.Detections,
    image,
    classes: list[str],
    p_out_binmask=Path('binmask.jpg'),
    p_out_seg=Path('seg.jpg'),
    verbose=False
):
    binary_mask = detections.mask[0].astype(np.uint8)*255
    cv2.imwrite(str(p_out_binmask), binary_mask)

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    cv2.imwrite(str(p_out_seg), annotated_image)
    if verbose:
        print(f'Binary mask annotation saved at: {str(p_out_binmask)}')
        print(f'Segmentation mask annotation saved at: {str(p_out_seg)}')


# for video
def load_table(p_csv):
    df = pd.read_csv(p_csv)
    return df


def get_vr_decord(p_video):
    vr = VideoReader(str(p_video), ctx=cpu(0))
    return vr


if __name__ == "__main__":
    args = parse_args()

    @dataclass
    class Items:
        OUT_DIR: str = args.OUT_DIR
        GROUNDING_DINO_CONFIG_PATH: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT_PATH: str = "./groundingdino_swint_ogc.pth"
        SAM_ENCODER_VERSION: str = "vit_h"
        SAM_CHECKPOINT_PATH: str = "./sam_vit_h_4b8939.pth"
        DEVICE: str = 'cuda:0'
        BOX_THRESHOLD: float = 0.25
        TEXT_THRESHOLD: float = 0.25
        NMS_THRESHOLD: float = 0.8
        VERBOSE: bool = False
        SAVE_IMAGE: bool = False

    args = Items()

    table = load_table('data/Ego4D/EgoNLQ/csvs/nlq_train.csv')
    clip_uids = list(set(table['clip_uid'].tolist()))
    print(f'# Clips: {len(clip_uids)}')
    # main(args, clip_uids[0])
    jobs = submit_job_to_slurm(args, clip_uids)
    for job in jobs:
        print(f"Submitted job with ID: {job.job_id}")
    for job in jobs:
        job.result()
