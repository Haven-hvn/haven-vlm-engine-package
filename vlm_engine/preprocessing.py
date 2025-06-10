import sys
import platform
import torch
from torchvision.io import read_image, VideoReader
import torchvision
from typing import List, Tuple, Dict, Any, Optional, Union, Iterator
import logging
from PIL import Image as PILImage

is_macos_arm = sys.platform == 'darwin' and platform.machine() == 'arm64'

if is_macos_arm:
    import av
else:
    import decord
    decord.bridge.set_bridge('torch')

def custom_round(number: float) -> int:
    if number - int(number) >= 0.5:
        return int(number) + 1
    else:
        return int(number)

def get_video_duration_decord(video_path: str) -> float:
    try:
        if is_macos_arm:
            container = av.open(video_path)
            if not container.streams.video:
                return 0.0
            stream = container.streams.video[0]
            if stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            else:
                fps = stream.average_rate
                if fps and stream.frames:
                    duration = float(stream.frames / float(fps))
                else:
                    duration = 0.0
            container.close()
            return duration
        else:
            vr: decord.VideoReader = decord.VideoReader(video_path, ctx=decord.cpu(0))
            num_frames: int = len(vr)
            frame_rate: float = vr.get_avg_fps()
            if frame_rate == 0: return 0.0
            duration: float = num_frames / frame_rate
            del vr
            return duration
    except Exception as e:
        logging.getLogger("logger").error(f"Error reading video {video_path}: {e}")
        return 0.0

def preprocess_video(video_path: str, frame_interval_sec: float = 0.5, img_size: Union[int, Tuple[int,int]] = 512, use_half_precision: bool = True, device_str: Optional[str] = None, use_timestamps: bool = False, vr_video: bool = False, norm_config_idx: int = 1, process_for_vlm: bool = False) -> Iterator[Tuple[Union[int, float], torch.Tensor]]:
    actual_device: torch.device = torch.device(device_str) if device_str else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = logging.getLogger("logger")

    if is_macos_arm:
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            if fps == 0:
                logger.warning(f"Video {video_path} has FPS of 0. Cannot process.")
                container.close()
                return
            
            frames_to_skip = custom_round(fps * frame_interval_sec)
            if frames_to_skip < 1: frames_to_skip = 1
            
            frame_count = 0
            for frame in container.decode(stream):
                if frame_count % frames_to_skip == 0:
                    frame_np = frame.to_ndarray(format='rgb24')
                    frame_tensor = torch.from_numpy(frame_np).to(actual_device)
                    
                    if process_for_vlm:
                        frame_tensor = crop_black_bars_lr(frame_tensor)
                        
                        if not torch.is_floating_point(frame_tensor):
                            frame_tensor = frame_tensor.float()
                        
                        if use_half_precision:
                            frame_tensor = frame_tensor.half()
                        
                        transformed_frame = frame_tensor
                        frame_identifier = frame_count / fps if use_timestamps else frame_count
                        yield (frame_identifier, transformed_frame)
                    else:
                        logger.warning("Standard processing path no longer supported - use VLM processing")
                        continue
                
                frame_count += 1
            
            container.close()
            
        except Exception as e:
            logger.error(f"PyAV failed to process video {video_path}: {e}")
            return
    else:
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        except RuntimeError as e:
            logger.error(f"Decord failed to open video {video_path}: {e}")
            return
            
        fps: float = vr.get_avg_fps()
        if fps == 0:
            logger.warning(f"Video {video_path} has FPS of 0. Cannot process.")
            if 'vr' in locals(): del vr
            return

        frames_to_skip: int = custom_round(fps * frame_interval_sec) 
        if frames_to_skip < 1: frames_to_skip = 1

        if process_for_vlm:
            for i in range(0, len(vr), frames_to_skip):
                try:
                    frame_cpu = vr[i] 
                except RuntimeError as e_read_frame:
                    logger.warning(f"Could not read frame {i} from {video_path}: {e_read_frame}")
                    continue
                # Convert decord NDArray to PyTorch tensor if needed
                if not isinstance(frame_cpu, torch.Tensor):
                    frame_cpu = torch.from_numpy(frame_cpu.asnumpy())
                    
                frame_cpu = crop_black_bars_lr(frame_cpu)
                frame = frame_cpu.to(actual_device)
                
                if not torch.is_floating_point(frame):
                    frame = frame.float()
                
                if use_half_precision:
                    frame = frame.half()
                
                transformed_frame = frame
                
                frame_identifier: Union[int, float] = i / fps if use_timestamps else i
                yield (frame_identifier, transformed_frame)
        else:
            logger.warning("Standard processing path no longer supported - use VLM processing")
            return
                
        if 'vr' in locals(): del vr

def crop_black_bars_lr(frame: torch.Tensor, black_threshold: float = 10.0, column_black_pixel_fraction_threshold: float = 0.95) -> torch.Tensor:
    logger = logging.getLogger("logger")
    if not isinstance(frame, torch.Tensor) or frame.ndim != 3 or frame.shape[2] < 3:
        logger.warning(f"crop_black_bars_lr: Invalid frame shape {frame.shape if isinstance(frame, torch.Tensor) else type(frame)}, returning original frame.")
        return frame

    H, W, C = frame.shape
    if W == 0 or H == 0:
        logger.debug("crop_black_bars_lr: Frame has zero width or height, returning original frame.")
        return frame

    rgb_frame = frame[:, :, :3]
    is_black_pixel = torch.all(rgb_frame < black_threshold, dim=2)
    column_black_pixel_count = torch.sum(is_black_pixel, dim=0)
    column_black_fraction = column_black_pixel_count.float() / H
    is_black_bar_column = column_black_fraction >= column_black_pixel_fraction_threshold

    x_start = 0
    for i in range(W):
        if not is_black_bar_column[i]:
            x_start = i
            break
    else:
        logger.debug("crop_black_bars_lr: Frame appears to be entirely black or too narrow. No crop applied.")
        return frame

    x_end = W
    for i in range(W - 1, x_start -1, -1):
        if not is_black_bar_column[i]:
            x_end = i + 1
            break
    
    if x_start >= x_end:
        logger.warning(f"crop_black_bars_lr: Inconsistent crop boundaries (x_start={x_start}, x_end={x_end}). No crop applied.")
        return frame
    
    if x_start == 0 and x_end == W:
        return frame

    cropped_frame = frame[:, x_start:x_end, :]
    logger.debug(f"Cropped frame from W={W} to W'={cropped_frame.shape[1]} (x_start={x_start}, x_end={x_end})")
    return cropped_frame.clone()
