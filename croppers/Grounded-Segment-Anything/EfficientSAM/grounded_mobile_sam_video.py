import cv2
import numpy as np
import supervision as sv
import argparse
import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import SamPredictor
from MobileSAM.setup_mobile_sam import setup_model

# for video
import time
from pathlib import Path
from pprint import pprint
import pandas as pd
from decord import VideoReader, cpu


def main():
    table = load_table('data/Ego4D/EgoNLQ/csvs/nlq_train.csv')
    p_clip_dir = Path('/data/datasets/ego4d_data/v2/clips_320p-non_official')
    clip_uids = set(table['clip_uid'].tolist())

    # debug
    sample_uid = sorted(clip_uids)[3]
    p_clip = p_clip_dir / f'{sample_uid}.mp4'
    assert p_clip.exists(), p_clip
    p_out_dir = OUT_DIR or Path(f'tmp/gmsam_sample_results/{sample_uid}/')
    p_out_dir.mkdir(exist_ok=True, parents=True)

    grounding_dino_model = load_gdino(DEVICE)
    sam_predictor = load_sam(MOBILE_SAM_CHECKPOINT_PATH, DEVICE)

    vr = get_vr_decord(p_clip)
    video_BGR = vr[:].asnumpy()[..., ::-1]
    clip_table = table[table["clip_uid"]==sample_uid]
    # slot_x or slot_y or verb_x or verb_y or query
    classes_sr = clip_table['slot_x'].fillna(clip_table['slot_y']).fillna(clip_table['verb_x']).fillna(clip_table['verb_y']).fillna(clip_table['query'])
    assert not (na_idx:=classes_sr.isna()).any(), f'At least one of lines has a nan:\n\t{table[na_idx]}'
    classes = classes_sr.tolist()

    def segment_image(image_BGR: np.ndarray, classes: list[str]) -> sv.Detections:
        detections = detect_image(grounding_dino_model, image_BGR, classes=classes)
        detections = nms(detections, VERBOSE)
        detections = segment(sam_predictor, detections, image_BGR)
        return detections

    def segment_video(video_BGR: np.ndarray, classes: list[str]):
        video_detections = []

        if SAVE_IMAGE:
            p_out_box_dir, p_out_mask_dir, p_out_seg_dir = (
                p_out_dir / 'box',
                p_out_dir / 'mask',
                p_out_dir / 'seg'
            )
            for p_dir in (p_out_box_dir, p_out_mask_dir, p_out_seg_dir):
                p_dir.mkdir(exist_ok=True, parents=True)

        t0 = time.time()
        for frame_idx, frame_BGR in enumerate(video_BGR):
            frame_detections = segment_image(frame_BGR, classes)
            if VERBOSE:
                for xyxy, mask, confidence, class_id, tracker_id in frame_detections:
                    print(f'name: {classes[class_id]}')
                    print(f'\txyxy: {xyxy}')
                    print(f'\tconf: {confidence}')
                    print(f'\tidx : {class_id}')
                    print()
            else:
                dt = time.time() - t0
                t_pf = dt / (frame_idx + 1)
                print(f'[{frame_idx:06d} / {video_BGR.shape[0]}] [{dt:.3f} s | {t_pf:.3f} s/frame]')

            if SAVE_IMAGE:
                p_out_box, p_out_mask, p_out_seg = (
                    p_out_box_dir / f'{frame_idx:06d}.jpg',
                    p_out_mask_dir / f'{frame_idx:06d}.jpg',
                    p_out_seg_dir / f'{frame_idx:06d}.jpg'
                )
                annotate_bbox(frame_detections, frame_BGR, classes, p_out_box)
                annotate_masks(frame_detections, frame_BGR, classes, p_out_mask, p_out_seg)
            video_detections.append(frame_detections)
        return video_detections

    print(clip_table.shape[0], classes)
    detections = segment_video(video_BGR, classes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--MOBILE_SAM_CHECKPOINT_PATH", type=str, default="./EfficientSAM/mobile_sam.pt", help="model"
    )
    parser.add_argument(
        "--WORKER_ID", type=int, default=0, help="worker id"
    )
    parser.add_argument(
        "--OUT_FILE_BOX", type=str, default="groundingdino_annotated_image.jpg", help="the output filename"
    )
    parser.add_argument(
        "--OUT_DIR", type=str, default=None, help="the output dirname"
    )
    parser.add_argument("--BOX_THRESHOLD", type=float, default=0.25, help="")
    parser.add_argument("--TEXT_THRESHOLD", type=float, default=0.25, help="")
    parser.add_argument("--NMS_THRESHOLD", type=float, default=0.8, help="")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--DEVICE", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    return parser.parse_args()


def load_gdino(device='cuda'):
    grounding_dino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        device=device,
    )
    return grounding_dino_model


def load_sam(p_ckpt, device='cuda'):
    checkpoint = torch.load(p_ckpt)
    mobile_sam = setup_model()
    mobile_sam.load_state_dict(checkpoint, strict=True)
    mobile_sam.to(device=device)

    sam_predictor = SamPredictor(mobile_sam)
    return sam_predictor


def read_image_cv2(p_img):
    image = cv2.imread(p_img)
    return image


def detect_image(detection_model: Model, image_BGR, classes: list[str]):
    detections = detection_model.predict_with_classes(
        image=image_BGR,
        classes=classes,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    return detections


def nms(detections: sv.Detections, VERBOSE=False):
    if VERBOSE:
        print(f"Before NMS: {len(detections.xyxy)} boxes")

    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        NMS_THRESHOLD
    ).numpy().tolist()
    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    if VERBOSE:
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
    VERBOSE=False
):
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    cv2.imwrite(str(p_out_box), annotated_frame)
    if VERBOSE:
        print(f'Bbox annotation saved at: {str(p_out_box)}')


def annotate_masks(
    detections: sv.Detections,
    image,
    classes: list[str],
    p_out_binmask=Path('binmask.jpg'),
    p_out_seg=Path('seg.jpg'),
    VERBOSE=False
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
    if VERBOSE:
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
    GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"
    MOBILE_SAM_CHECKPOINT_PATH = args.MOBILE_SAM_CHECKPOINT_PATH
    DEVICE = args.DEVICE
    WORKER_ID = args.WORKER_ID
    BOX_THRESHOLD = args.BOX_THRESHOLD
    TEXT_THRESHOLD = args.TEXT_THRESHOLD
    NMS_THRESHOLD = args.NMS_THRESHOLD
    OUT_DIR = args.OUT_DIR
    VERBOSE = False
    SAVE_IMAGE = False
    main()
