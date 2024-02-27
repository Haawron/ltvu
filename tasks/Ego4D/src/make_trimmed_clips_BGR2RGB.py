import cv2
import os
from pathlib import Path
from multiprocessing import Pool


def process_video(mp4_path, p_mp4_dir_tgt):
    """
    Process a single video file: read it, change BGR to RGB, and write to a new directory.

    :param mp4_path: Path to the original MP4 file.
    :param p_mp4_dir_tgt: Directory where the processed video will be saved.
    """
    # Create a VideoCapture object
    cap = cv2.VideoCapture(str(mp4_path))

    # Check if video opened successfully
    if not cap.isOpened():
        print(f'Error opening video file {mp4_path}')
        return
    else:
        print(f'Start processing {mp4_path}')

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(p_mp4_dir_tgt / mp4_path.name), fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[..., ::-1]
        out.write(frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    print(f'Done processing {mp4_path}')


def parallel_process(p_mp4s, p_mp4_dir_tgt, num_workers):
    """
    Process the list of MP4 paths in parallel.

    :param p_mp4s: List of paths to MP4 files.
    :param p_mp4_dir_tgt: Directory where the processed videos will be saved.
    :param num_workers: Number of worker processes to use.
    """
    p_mp4_dir_tgt = Path(p_mp4_dir_tgt)
    p_mp4_dir_tgt.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    with Pool(num_workers) as pool:
        pool.starmap(process_video, [(p_mp4s[i], p_mp4_dir_tgt) for i in range(len(p_mp4s))])


if __name__ == '__main__':
    p_mp4_dir_src = Path('/data/datasets/ego4d_data/v2/tmp-clips_320p-non_official/')
    p_mp4_dir_tgt = Path('/data/datasets/ego4d_data/v2/clips_320p-non_official/')
    p_mp4s = sorted(p_mp4_dir_src.glob('*.mp4'))
    num_workers = os.cpu_count()

    # Call the parallel processing function
    parallel_process(p_mp4s, p_mp4_dir_tgt, num_workers)
