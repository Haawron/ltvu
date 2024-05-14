import logging
import subprocess
from pathlib import Path

from ltvu.objects import *


logger = logging.getLogger(__name__)


def get_video_fps(p_video):
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'csv=s=x:p=0',
        str(p_video)]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    try:
        fps = eval(result.stdout)
    except (ValueError, SyntaxError):
        raise ValueError("Could not retrieve video FPS.")
    return fps


def get_video_length(p_video):
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(p_video)]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    try:
        duration = float(result.stdout)
    except ValueError:
        raise ValueError("Could not retrieve video duration.")
    return duration


def trim_video(
    p_video: Path,
    s: TimestampType|IntervalType,
    e: TimestampType|None = None,
    pass_if_exists = False
):
    if not p_video.exists():
        raise FileNotFoundError(f'{p_video} does not exist.')
    duration_sec = get_video_length(p_video)
    FPS = get_video_fps(p_video)
    itvl = Interval(s, e, idxs_per_sec=FPS)
    s, e = itvl.s, itvl.e
    eps = 1e-6

    if s.sec < 0:
        raise ValueError(f'Start time {s.sec}s is less than 0.')
    if e.sec + eps > duration_sec:
        raise ValueError(f'End time {e.sec}s is greater than video duration {duration_sec:.1f}s.')

    p_splitted_video_dir = Path('/tmp/video-llava')
    p_splitted_video = p_splitted_video_dir / f'{p_video.stem}_{s.sec:.1f}_{e.sec:.1f}{p_video.suffix}'
    if pass_if_exists and p_splitted_video.exists():
        return p_splitted_video

    p_splitted_video_dir.mkdir(exist_ok=True, parents=True)
    cmd = [
        'ffmpeg',
        '-i', str(p_video),
        '-ss', str(s.sec),
        '-t', str(itvl.l.sec),
        '-an',  # removes the audio channel
        '-c', 'copy', '-avoid_negative_ts', '1',
        '-y',
        str(p_splitted_video)
    ]
    # logger.debug('\n' + ' '.join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.debug(str(p_splitted_video))
    return p_splitted_video


if __name__ == '__main__':
    p_sample_video = Path('/data/datasets/ego4d_data/v2/clips_320p-non_official/0aca0078-b6ab-41fb-9dc5-a70b8ad137b2.mp4')
    duration_sec = get_video_length(p_sample_video)  # 480.034
    print(duration_sec)
    p_splitted_video = trim_video(p_sample_video, Interval(30., 50.))
    print(p_splitted_video)  # /tmp/video-llava/0aca0078-b6ab-41fb-9dc5-a70b8ad137b2_30.0_50.0.mp4
