import sys
import platform
import torch
from torchvision.transforms import v2 as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.io import read_image, VideoReader
import torchvision
from typing import List, Tuple, Dict, Any, Optional, Union, Iterator
import logging
from PIL import Image as PILImage

# Platform-specific imports
is_macos_arm = sys.platform == 'darwin' and platform.machine() == 'arm64'

if is_macos_arm:
    try:
        import av
    except ImportError:
        av = None
        logging.getLogger("logger").warning("PyAV not installed. Video processing on macOS ARM might be limited.")
else:
    try:
        import decord
        decord.bridge.set_bridge('torch')
    except ImportError:
        decord = None
        logging.getLogger("logger").warning("Decord not installed. Video processing on non-macOS ARM might be limited.")

def custom_round(number: float) -> int:
    if number - int(number) >= 0.5:
        return int(number) + 1
    else:
        return int(number)

def get_normalization_config(config_index: Union[int, Dict[str, List[float]]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(config_index, int) and config_index not in [1, 2, 3]:
        config_index = 1

    if isinstance(config_index, dict):
        mean_values = config_index.get("mean", [0.485, 0.456, 0.406])
        std_values = config_index.get("std", [0.229, 0.224, 0.225])
        mean = torch.tensor(mean_values, device=device, dtype=torch.float32)
        std = torch.tensor(std_values, device=device, dtype=torch.float32)
    elif config_index == 1:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32)
    elif config_index == 2:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device, dtype=torch.float32)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=torch.float32)
    elif config_index == 3:
        mean = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        std = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32)
    return mean, std

def get_video_duration_torchvision(video_path: str) -> float:
    try:
        video: VideoReader = torchvision.io.VideoReader(video_path, "video")
        metadata: Dict[str, Any] = video.get_metadata()
        duration: float = 0.0
        if metadata and 'video' in metadata and metadata['video']['duration'] and metadata['video']['duration'][0]:
            duration = float(metadata['video']['duration'][0])
        del video
        return duration
    except Exception as e:
        logging.getLogger("logger").error(f"Torchvision VideoReader failed for {video_path}: {e}")
        return 0.0


def get_video_duration_decord(video_path: str) -> float:
    logger = logging.getLogger("logger")
    try:
        if is_macos_arm:
            if not av: raise RuntimeError("PyAV is not installed or failed to import.")
            container = av.open(video_path)
            if not container.streams.video:
                container.close()
                return 0.0
            stream = container.streams.video[0]
            duration = 0.0
            if stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            elif stream.frames and stream.average_rate: # Fallback for some formats
                 duration = float(stream.frames / float(stream.average_rate))
            container.close()
            return duration
        else:
            if not decord: raise RuntimeError("Decord is not installed or failed to import.")
            vr: decord.VideoReader = decord.VideoReader(video_path, ctx=decord.cpu(0))
            num_frames: int = len(vr)
            frame_rate: float = vr.get_avg_fps()
            if frame_rate == 0: return 0.0
            duration: float = num_frames / frame_rate
            del vr
            return duration
    except Exception as e:
        logger.error(f"Error reading video duration for {video_path}: {e}")
        return 0.0

def get_frame_transforms(use_half_precision: bool, mean: torch.Tensor, std: torch.Tensor, vr_video: bool, img_size: Union[int, Tuple[int, int]]) -> transforms.Compose:
    normalize_transform = transforms.Normalize(mean=mean, std=std)
    transform_list = []

    if isinstance(img_size, int):
        target_size = (img_size, img_size)
    elif isinstance(img_size, tuple) and len(img_size) == 2:
        target_size = img_size
    else:
        logging.getLogger("logger").warning(f"Invalid img_size {img_size}, defaulting to 224x224 for transforms.")
        target_size = (224, 224)

    transform_list.extend([
        transforms.ToDtype(torch.half if use_half_precision else torch.float32, scale=True),
        transforms.Resize(target_size, interpolation=InterpolationMode.BICUBIC, antialias=True), # type: ignore
        normalize_transform
    ])
    return transforms.Compose(transform_list)

def vr_permute(frame: torch.Tensor) -> torch.Tensor:
    logger = logging.getLogger("logger")
    if frame.ndim != 3 or frame.shape[2] != 3:
        logger.warning(f"vr_permute received unexpected frame shape: {frame.shape}. Expecting HWC format.")
        return frame

    height: int = frame.shape[0]
    width: int = frame.shape[1]
    if height == 0 or width == 0: return frame

    aspect_ratio: float = width / height
    if aspect_ratio > 1.5:
        return frame[:, :width//2, :]
    else:
        return frame[:height//2, width//4:(width//4 + width//2), :]

def preprocess_image_from_path(image_path_or_pil: Union[str, PILImage.Image, torch.Tensor], img_size: Union[int, Tuple[int,int]] = 512, use_half_precision: bool = True, device_str: Optional[str] = None, norm_config_idx: Union[int, Dict[str, List[float]]] = 1) -> torch.Tensor:
    actual_device: torch.device = torch.device(device_str) if device_str else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean, std = get_normalization_config(norm_config_idx, actual_device)
    target_dtype: torch.dtype = torch.float16 if use_half_precision and actual_device.type == 'cuda' else torch.float32

    img_tensor: torch.Tensor
    if isinstance(image_path_or_pil, str):
        img_tensor = read_image(image_path_or_pil).to(actual_device)
    elif isinstance(image_path_or_pil, PILImage.Image):
        img_tensor = transforms.functional.pil_to_tensor(image_path_or_pil).to(actual_device)
    elif isinstance(image_path_or_pil, torch.Tensor):
        img_tensor = image_path_or_pil.to(actual_device)
    else:
        raise TypeError(f"Unsupported input type for preprocess_image: {type(image_path_or_pil)}")

    if img_tensor.ndim == 2: img_tensor = img_tensor.unsqueeze(0)
    if img_tensor.shape[0] == 1: img_tensor = img_tensor.repeat(3, 1, 1)
    elif img_tensor.shape[0] == 4: img_tensor = img_tensor[:3, :, :]
    
    current_img_size_for_resize: Tuple[int,int] = (img_size, img_size) if isinstance(img_size, int) else img_size

    transform_list_img: List[Any] = [
        transforms.Resize(current_img_size_for_resize, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.ToDtype(target_dtype, scale=True),
        transforms.Normalize(mean=mean, std=std)
    ]
    image_transforms_comp: transforms.Compose = transforms.Compose(transform_list_img)
    return image_transforms_comp(img_tensor)

def preprocess_video(video_path: str, frame_interval_sec: float = 0.5, img_size: Union[int, Tuple[int,int]] = 512, use_half_precision: bool = True, device_str: Optional[str] = None, use_timestamps: bool = False, vr_video: bool = False, norm_config_idx: Union[int, Dict[str, List[float]]] = 1, process_for_vlm: bool = False) -> Iterator[Tuple[Union[int, float], torch.Tensor]]:
    actual_device: torch.device = torch.device(device_str) if device_str else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = logging.getLogger("logger")

    if is_macos_arm:
        if not av:
            logger.error("PyAV not available, cannot process video on macOS ARM.")
            return
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            if not stream: 
                logger.error(f"No video stream found in {video_path}")
                container.close()
                return

            fps = float(stream.average_rate) if stream.average_rate else float(stream.guessed_rate) # Fallback to guessed_rate
            if fps == 0:
                logger.warning(f"Video {video_path} has FPS of 0. Cannot process.")
                container.close()
                return
            
            frames_to_skip = custom_round(fps * frame_interval_sec)
            if frames_to_skip < 1: frames_to_skip = 1
            
            frame_count = 0
            for frame_av in container.decode(stream):
                if frame_count % frames_to_skip == 0:
                    frame_np = frame_av.to_ndarray(format='rgb24') # HWC, uint8
                    frame_tensor = torch.from_numpy(frame_np).to(actual_device) # HWC, uint8
                    
                    if process_for_vlm:
                        frame_tensor = crop_black_bars_lr(frame_tensor) # HWC, uint8
                        frame_tensor = frame_tensor.permute(2, 0, 1) # CHW, uint8
                        frame_tensor = transforms.functional.to_dtype(frame_tensor, torch.float32 if not use_half_precision or actual_device.type == 'cpu' else torch.float16, scale=True) # CHW, float/half, [0,1]
                        if vr_video: # vr_permute expects HWC, so permute back, then permute again after
                            frame_tensor = frame_tensor.permute(1, 2, 0) # HWC
                            frame_tensor = vr_permute(frame_tensor)    # HWC
                            frame_tensor = frame_tensor.permute(2, 0, 1) # CHW
                        transformed_frame = frame_tensor
                    else:
                        mean, std = get_normalization_config(norm_config_idx, actual_device)
                        frame_transforms_comp = get_frame_transforms(use_half_precision and actual_device.type == 'cuda', mean, std, vr_video, img_size)
                        
                        if vr_video: frame_tensor = vr_permute(frame_tensor) # HWC
                        frame_tensor = frame_tensor.permute(2, 0, 1) # CHW, uint8
                        transformed_frame = frame_transforms_comp(frame_tensor) # CHW, float/half, normalized
                        
                    frame_identifier = frame_count / fps if use_timestamps else frame_count
                    yield (frame_identifier, transformed_frame)
                frame_count += 1
            container.close()
        except Exception as e:
            logger.error(f"PyAV failed to process video {video_path}: {e}", exc_info=True)
            return
    else: # Decord path
        if not decord:
            logger.error("Decord not available, cannot process video.")
            return
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        except RuntimeError as e:
            logger.error(f"Decord failed to open video {video_path}: {e}")
            return
            
        fps: float = vr.get_avg_fps()
        if fps == 0:
            logger.warning(f"Video {video_path} has FPS of 0. Cannot process.")
            del vr
            return

        frames_to_skip: int = custom_round(fps * frame_interval_sec)
        if frames_to_skip < 1: frames_to_skip = 1

        for i in range(0, len(vr), frames_to_skip):
            try:
                frame_tensor = vr[i].to(actual_device) # HWC, uint8, on target device
            except RuntimeError as e_read_frame:
                logger.warning(f"Could not read frame {i} from {video_path}: {e_read_frame}")
                continue
            
            if process_for_vlm:
                frame_tensor = crop_black_bars_lr(frame_tensor) # HWC, uint8
                frame_tensor = frame_tensor.permute(2, 0, 1) # CHW, uint8
                frame_tensor = transforms.functional.to_dtype(frame_tensor, torch.float32 if not use_half_precision or actual_device.type == 'cpu' else torch.float16, scale=True) # CHW, float/half, [0,1]
                if vr_video:
                    frame_tensor = frame_tensor.permute(1, 2, 0) # HWC
                    frame_tensor = vr_permute(frame_tensor)    # HWC
                    frame_tensor = frame_tensor.permute(2, 0, 1) # CHW
                transformed_frame = frame_tensor
            else:
                mean, std = get_normalization_config(norm_config_idx, actual_device)
                frame_transforms_comp = get_frame_transforms(use_half_precision and actual_device.type == 'cuda', mean, std, vr_video, img_size)
                if vr_video: frame_tensor = vr_permute(frame_tensor) # HWC
                frame_tensor = frame_tensor.permute(2, 0, 1) # CHW, uint8
                transformed_frame = frame_transforms_comp(frame_tensor) # CHW, float/half, normalized
                
            frame_identifier: Union[int, float] = i / fps if use_timestamps else i
            yield (frame_identifier, transformed_frame)
        del vr

def crop_black_bars_lr(frame: torch.Tensor, black_threshold: float = 10.0, column_black_pixel_fraction_threshold: float = 0.95) -> torch.Tensor:
    logger = logging.getLogger("logger")
    if not isinstance(frame, torch.Tensor) or frame.ndim != 3 or frame.shape[0] < 3 : # Expect HWC (C,H,W after permute)
        # If CHW, permute to HWC for this logic
        if isinstance(frame, torch.Tensor) and frame.ndim == 3 and frame.shape[0] == 3 : # CHW
             frame_hwc = frame.permute(1,2,0)
        else:
            logger.warning(f"crop_black_bars_lr: Invalid frame shape {frame.shape if isinstance(frame, torch.Tensor) else type(frame)}, returning original frame.")
            return frame
    else: # Already HWC
        frame_hwc = frame

    H, W, C = frame_hwc.shape
    if W == 0 or H == 0: return frame_hwc.clone() if frame_hwc is frame else frame_hwc # return original if no change

    rgb_frame = frame_hwc[:, :, :3]
    is_black_pixel = torch.all(rgb_frame < black_threshold, dim=2)
    column_black_pixel_count = torch.sum(is_black_pixel, dim=0)
    column_black_fraction = column_black_pixel_count.float() / H
    is_black_bar_column = column_black_fraction >= column_black_pixel_fraction_threshold

    x_start = 0
    for i in range(W):
        if not is_black_bar_column[i]:
            x_start = i
            break
    else: return frame_hwc.clone() if frame_hwc is frame else frame_hwc


    x_end = W
    for i in range(W - 1, x_start -1, -1):
        if not is_black_bar_column[i]:
            x_end = i + 1
            break
    
    if x_start >= x_end: return frame_hwc.clone() if frame_hwc is frame else frame_hwc
    if x_start == 0 and x_end == W: return frame_hwc.clone() if frame_hwc is frame else frame_hwc

    cropped_frame_hwc = frame_hwc[:, x_start:x_end, :].clone()
    logger.debug(f"Cropped frame from W={W} to W'={cropped_frame_hwc.shape[1]} (x_start={x_start}, x_end={x_end})")
    
    # If original was CHW, permute back
    if isinstance(frame, torch.Tensor) and frame.ndim == 3 and frame.shape[0] == 3 and frame_hwc is not frame : # CHW input
        return cropped_frame_hwc.permute(2,0,1)
    return cropped_frame_hwc
