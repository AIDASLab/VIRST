import random
import os
import io
import math
import re
import gc

import av
import cv2
import decord
import imageio
from decord import VideoReader
import numpy as np
from typing import List 
from PIL import Image as PILImage


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def lazy_load_s3video(s3path_video, num_frames, video_start, video_end, client):
    video_bytes_stream = client.get(s3path_video, enable_stream_lazyloding=True)
    container = av.open(video_bytes_stream)
    stream = container.streams.video[0]
    real_fps = container.streams.video[0].average_rate
    time_base = container.streams.video[0].time_base
    start, end = video_start, video_end
    duration_frams = int(end - start) * real_fps
    frames_index = get_index(duration_frams, num_frames)

    pts_list = []

    start_pts = int((start) / time_base)
    end_pts = int((end) / time_base)
    for frame_index in frames_index:
        pts_list.append(int((frame_index / real_fps)) /  time_base)

    container.seek(max(start_pts, 0), stream=stream)
    
    frames = []
    for frame in container.decode(**{"video":0}):
        if frame.pts < start_pts:
            continue
        if len(pts_list) >0:
            if frame.pts >= pts_list[0]:
                frames.append(frame)
                pts_list.pop(0)
        else:
            break
    container.close()
    frames = [np.array(frames[idx].to_rgb().to_image()) for idx in range(len(frames))]
    final_frames = np.stack(frames)
    del frames
    del video_bytes_stream
    
    gc.collect()
    
    return final_frames, frames_index, float(real_fps)

def pts_to_secs(pts: int, time_base: float, start_pts: int) -> float:
    """
    Converts a present time with the given time base and start_pts offset to seconds.

    Returns:
        time_in_seconds (float): The corresponding time in seconds.

    https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/utils.py#L54-L64
    """
    if pts == math.inf:
        return math.inf

    return int(pts - start_pts) * time_base


def get_pyav_video_duration(video_reader):
    video_stream = video_reader.streams.video[0]
    video_duration = pts_to_secs(
        video_stream.duration,
        video_stream.time_base,
        video_stream.start_time
    )
    return float(video_duration)



def get_frame_indices(num_frames, vlen, sample='middle', fix_start=None, input_fps=1, min_num_frames=1, max_num_frames=-1, local_num_frames=8):

    if min_num_frames > vlen:
        if sample == 'dynamic_fps1':
            min_num_frames = (vlen // local_num_frames) * local_num_frames
        else:
            min_num_frames = vlen


    if sample == 'dynamic_fps1':

        duration = float(vlen) / input_fps
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments

        if max_num_frames > 0:
            num_frames = min(num_frames, max_num_frames)
        sample = "middle" # NOTE

    num_frames = max(min_num_frames, num_frames)

    if sample in ["rand", "middle"]:
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
    else:
        raise ValueError(f"Not support sample type: {sample}")
    
    
    return frame_indices


def read_frames_av(video_path, num_frames, sample='rand', client=None, fix_start=None, min_num_frames=1, max_num_frames=-1, clip=None, local_num_frames=8):
    if clip is not None:
        raise NotImplementedError("av don't support clip!!!")
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        byteio = io.BytesIO(video_bytes)
        byteio.seek(0)
        reader = av.open(byteio)
    else:
        byteio = None
        reader = av.open(video_path)
    frames = [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]
    vlen = len(frames)
    duration = get_pyav_video_duration(reader)
    fps = vlen / float(duration)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, min_num_frames=min_num_frames, max_num_frames=max_num_frames, local_num_frames=local_num_frames
    )
    frames = np.stack([frames[idx] for idx in frame_indices])
    if byteio is not None:
        byteio.close()
        
    reader.close()

    return frames, frame_indices, float(fps), duration


def read_frames_gif(
        video_path, num_frames, sample='rand', fix_start=None, 
        min_num_frames=1, max_num_frames=-1, client=None, clip=None, local_num_frames=8
    ):
    if clip is not None:
        raise NotImplementedError("Gif don't support clip!!!")
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        byteio = io.BytesIO(video_bytes)
        gif = imageio.get_reader(byteio)
    else:
        byteio = None
        gif = imageio.get_reader(video_path)
    vlen = len(gif)
    fps = 1.
    duration = vlen / fps
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        min_num_frames=min_num_frames,
        max_num_frames=max_num_frames, local_num_frames=local_num_frames,
        input_fps=fps
    )
    frames = []

    min_h = min_w = 100000
    hw_set = set()
    for index, frame in enumerate(gif):
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = frame.astype(np.uint8)
            frames.append(frame)
            hw_set.add(frame.shape)
            if frame.shape[0] < min_h:
                min_h = frame.shape[0]
            if frame.shape[1] < min_w:
                min_w = frame.shape[1]
    if len(hw_set) > 1:
        frames = [i[:min_h, :min_w] for i in frames]

    frames = np.stack(frames)

    if byteio is not None:
        byteio.close()

    return frames, frame_indices, float(fps), duration



def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None, min_num_frames=1,
        max_num_frames=-1, client=None, clip=None, local_num_frames=8
    ):

    if video_path.endswith('.avi'):
        return read_frames_av(video_path=video_path, num_frames=num_frames, sample=sample,
                    fix_start=fix_start, min_num_frames=min_num_frames, max_num_frames=max_num_frames, 
                    client=client, clip=clip, local_num_frames=local_num_frames)
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        if video_bytes is None or len(video_bytes) == 0:
            raise ValueError(f"Can't read byte from {video_path}!")
        byteio = io.BytesIO(video_bytes)
        video_reader = VideoReader(byteio, num_threads=1)
    else:
        byteio = None
        video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    

    if clip:
        start, end = clip
        start = max(0, start)
        end = min(duration - 0.1, end)
        duration = end - start
        vlen = int(duration * fps) 
        start_index = int(start * fps)

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, min_num_frames=min_num_frames, max_num_frames=max_num_frames, local_num_frames=local_num_frames
    )
    if clip:
        frame_indices = [f + start_index for f in frame_indices]

    frames = video_reader.get_batch(frame_indices).asnumpy()
    video_reader.seek(0)

    if byteio is not None:
        byteio.close()
    return frames, frame_indices, float(fps), duration



def read_frames_img(
        video_path, num_frames, sample='rand', fix_start=None, min_num_frames=1,
        max_num_frames=-1, client=None, clip=None, local_num_frames=8
    ):
    def extract_frame_number(filename):
        if filename.endswith('.jpg'):
            match = re.search(r'_(\d+).jpg$', filename)
        elif filename.endswith('.jpeg'):
            match = re.search(r'_(\d+).jpeg$', filename)
        elif filename.endswith('.png'):
            match = re.search(r'_(\d+).png$', filename)
        else:
            raise NotImplementedError(f"Wrong filename: {filename}")

        return int(match.group(1)) if match else -1


    def sort_frames(frame_paths):
        return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))

    if "s3://" in video_path:
        img_list = sort_frames(client.list(video_path))
    else:
        img_list = sort_frames(list(os.listdir(video_path)))


    if 'tvqa' in video_path.lower():
        fps = 3.0
    else:
        fps = 1.0

    if clip is not None:
        start = float(clip[0])
        end = float(clip[1])
        start = max(0, start)
        end = min(len(img_list) / fps, end)
        vlen = (end - start) * fps
    else:
        vlen = len(img_list)
    
    duration = vlen / fps

    if min_num_frames > vlen:
        if sample == 'dynamic_fps1':
            min_num_frames = (vlen // local_num_frames) * local_num_frames
        else:
            min_num_frames = vlen

    if sample == 'dynamic_fps1':
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments
        num_frames = min(num_frames, max_num_frames) 
        num_frames = max(min_num_frames, num_frames)

    num_frames = int(num_frames)
    if clip is not None:
        def _get_index_by_time(start_sec, end_sec, num_segments=8, fps=1., max_frame=9999):
            start_idx = max(1, round(start_sec * fps))
            end_idx = min(round(end_sec * fps), max_frame)
            seg_size = float(end_idx - start_idx) / (num_segments - 1)
            offsets = np.array([start_idx + int(np.round(seg_size * idx)) for idx in range(num_segments)])
            return offsets

        frame_indices = _get_index_by_time(float(clip[0]), float(clip[1]), num_segments=num_frames, fps=fps, max_frame=len(img_list)-1)
    else:
        frame_indices = get_frame_indices(
            num_frames, vlen, sample=sample, fix_start=fix_start,
            min_num_frames=min_num_frames,
            max_num_frames=max_num_frames, local_num_frames=local_num_frames
        )

    imgs = []
    for idx in frame_indices:
        frame_fname = os.path.join(video_path, img_list[idx])
        if "s3://" in video_path:
            img_bytes = client.get(frame_fname)
        else:
            with open(frame_fname, 'rb') as f:
                img_bytes = f.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        imgs.append(img)

    frames = np.array(imgs, dtype=np.uint8)
    return frames, frame_indices, fps, duration



def read_frames_img_list(
        img_list:List[str], num_frames, sample='rand', fix_start=None, min_num_frames=1,
        max_num_frames=-1, client=None, clip=None, local_num_frames=8,
    ):
    def extract_frame_number(filename):
        if filename.endswith('.jpg'):
            match = re.search(r'_(\d+).jpg$', filename)
        elif filename.endswith('.jpeg'):
            match = re.search(r'_(\d+).jpeg$', filename)
        elif filename.endswith('.png'):
            match = re.search(r'_(\d+).png$', filename)
        else:
            raise NotImplementedError(f"Wrong filename: {filename}")

        return int(match.group(1)) if match else -1


    def sort_frames(frame_paths):
        return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))
    img_list = sort_frames(img_list)


    fps = 3.0

    if clip is not None:
        start = float(clip[0])
        end = float(clip[1])
        start = max(0, start)
        end = min(len(img_list) / fps, end)
        vlen = (end - start) * fps
    else:
        vlen = len(img_list)
    
    duration = vlen / fps

    num_frames = len(img_list)

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        min_num_frames=min_num_frames,
        max_num_frames=max_num_frames, local_num_frames=local_num_frames
    )

    imgs = []
    for idx in frame_indices:
        frame_fname = img_list[idx]
        
        with open(frame_fname, 'rb') as f:
            img_bytes = f.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        imgs.append(img)

    frames = np.array(imgs, dtype=np.uint8)

    return frames, frame_indices, fps, duration

def read_frames_sam2(
    img_list:List[str], frame_indices:List[str]):
    imgs = []
    for idx in frame_indices:
        frame_fname = img_list[idx] 
        with open(frame_fname, "rb") as f:
            imgs.append(PILImage.open(f).convert("RGB"))
    return imgs 

def sample_index_masks(
    mask_list:List[np.ndarray], frame_indices:List[int]
    ) -> List[np.ndarray]:
    masks = []
    for idx in frame_indices:
        if idx >= len(mask_list):
            mask = mask_list[-1]
        else:
            mask = mask_list[idx] 
        masks.append(mask)
    return masks

def read_frames_fake(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, clip=None, local_num_frames=8
    ):
    frame_indices = get_frame_indices(
        num_frames, 100, sample=sample, fix_start=fix_start,
        input_fps=1, max_num_frames=max_num_frames, local_num_frames=local_num_frames
    )
    frames = np.random.randint(0, 255, size=(len(frame_indices), 224, 224, 3)) # (T, H, W, C), torch.uint8
    return frames, frame_indices, 1.0, 100



VIDEO_READER_FUNCS = {
    'av': read_frames_av,
    'decord': read_frames_decord,
    'gif': read_frames_gif,
    'img': read_frames_img,
    'frame': read_frames_img,
    'lazy': lazy_load_s3video,
    'rvos': read_frames_img_list,
    'fake': read_frames_fake
}
