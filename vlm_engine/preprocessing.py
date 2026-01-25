import sys
import platform
import torch
from torchvision.io import read_image, VideoReader
import torchvision
from typing import List, Tuple, Dict, Any, Optional, Union, Iterator
import logging
import threading
from PIL import Image as PILImage

is_macos_arm = sys.platform == 'darwin' and platform.machine() == 'arm64'

try:
    import av  # type: ignore
except ImportError:
    av = None

# Global semaphore to limit concurrent PyAV operations since FFmpeg is not thread-safe
# This prevents issues when multiple threads try to decode videos simultaneously
# Using a semaphore with value 2 as a reasonable compromise between throughput and safety
_pyav_semaphore = threading.Semaphore(2)

def custom_round(number: float) -> int:
    if number - int(number) >= 0.5:
        return int(number) + 1
    else:
        return int(number)

def get_video_metadata(video_path: str, logger: Optional[logging.Logger] = None) -> Tuple[float, int]:
    """
    Get video FPS and total frames with robust error handling and fallback.
    
    Tries PyAV first (cross-platform), then falls back to decord.
    Provides detailed error diagnostics if both fail.
    
    Args:
        video_path: Path to the video file
        logger: Optional logger instance for logging
        
    Returns:
        Tuple of (fps, total_frames)
        
    Raises:
        ValueError: If video file is invalid or cannot be read by either library
    """
    import os
    
    if logger is None:
        logger = logging.getLogger("logger")
    
    # Validate file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    file_size = os.path.getsize(video_path)
    logger.info(f"Processing video: {video_path} ({file_size / 1024 / 1024:.2f} MB)")
    
    # Strategy 1: Try PyAV first (works on all platforms)
    if av is not None:
        try:
            # Use semaphore to limit concurrent PyAV operations
            with _pyav_semaphore:
                container = av.open(video_path, stream_options={'err_detect': 'ignore_err'})
                if not container.streams.video:
                    raise ValueError(f"No video stream found in: {video_path}")
                stream = container.streams.video[0]
                fps = float(stream.average_rate)
                total_frames = stream.frames if stream.frames > 0 else int(stream.duration * fps / stream.time_base)
                container.close()
            logger.info(f"Using PyAV for video metadata: {fps} fps, {total_frames} frames")
            return fps, total_frames
        except Exception as e:
            logger.debug(f"PyAV metadata extraction failed: {e}")
    
    # Strategy 2: Fallback to decord
    try:
        import decord  # type: ignore
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        fps = float(vr.get_avg_fps())
        total_frames = len(vr)
        del vr
        logger.info(f"Using decord for video metadata: {fps} fps, {total_frames} frames")
        return fps, total_frames
    except ImportError:
        raise ImportError("Neither PyAV nor decord available for video processing. Install decord with: pip install vlm-engine[decord]")
    except Exception as decord_error:
        logger.error(f"Decord failed to read video {video_path}: {decord_error}")
        
        # Provide detailed diagnostics
        error_details = []
        error_details.append(f"Video path: {video_path}")
        error_details.append(f"File exists: {os.path.exists(video_path)}")
        error_details.append(f"File size: {file_size} bytes")
        
        if os.path.exists(video_path):
            # Try to get basic file info
            with open(video_path, 'rb') as f:
                header = f.read(16)
                error_details.append(f"File header (hex): {header.hex()}")
        
        logger.error("; ".join(error_details))
        raise ValueError(
            f"Failed to read video with both PyAV and decord: {video_path}. "
            f"See debug logs for file diagnostics."
        ) from decord_error

def get_video_duration_decord(video_path: str) -> float:
    try:
        if is_macos_arm and av is not None:
            # Use semaphore to limit concurrent PyAV operations
            with _pyav_semaphore:
                container = av.open(video_path, stream_options={'err_detect': 'ignore_err'})
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
        elif is_macos_arm:
            # macOS ARM but av is not available, try decord
            try:
                import decord  # type: ignore
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
                num_frames = len(vr)
                frame_rate = vr.get_avg_fps()
                if frame_rate == 0: return 0.0
                duration = num_frames / frame_rate
                del vr
                return duration
            except ImportError:
                logging.getLogger("logger").error("Neither PyAV nor decord available on macOS ARM")
                return 0.0
        else:
            try:
                import decord  # type: ignore
                vr: decord.VideoReader = decord.VideoReader(video_path, ctx=decord.cpu(0))
                num_frames: int = len(vr)
                frame_rate: float = vr.get_avg_fps()
                if frame_rate == 0: return 0.0
                duration: float = num_frames / frame_rate
                del vr
                return duration
            except ImportError:
                logging.getLogger("logger").error("decord not available for video duration calculation")
                return 0.0
    except Exception as e:
        logging.getLogger("logger").error(f"Error reading video {video_path}: {e}")
        return 0.0

def preprocess_video(video_path: str, frame_interval_sec: float = 0.5, img_size: Union[int, Tuple[int,int]] = 512, use_half_precision: bool = True, use_timestamps: bool = False, vr_video: bool = False, norm_config_idx: int = 1, process_for_vlm: bool = False) -> Iterator[Tuple[Union[int, float], torch.Tensor]]:
    actual_device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = logging.getLogger("logger")

    if is_macos_arm and av is not None:
        container = None
        try:
            # Use semaphore to limit concurrent PyAV operations
            with _pyav_semaphore:
                container = av.open(video_path, stream_options={'err_detect': 'ignore_err'})
                stream = container.streams.video[0]
                fps = float(stream.average_rate)
                if fps == 0:
                    logger.warning(f"Video {video_path} has FPS of 0. Cannot process.")
                    if container:
                        container.close()
                    return
                
                frames_to_skip = custom_round(fps * frame_interval_sec)
                if frames_to_skip < 1: frames_to_skip = 1
                
                MAX_TOTAL_FRAMES = 10000  # Safety limit for corrupted videos
                frame_count = 0
                
                try:
                    decode_context = container.decode(stream)
                    for frame in decode_context:
                        if frame_count >= MAX_TOTAL_FRAMES:
                            logger.warning(f"Hit decode limit {MAX_TOTAL_FRAMES}, possible corruption in {video_path}")
                            break
                            
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
                                # Skip this frame but continue processing
                        
                        frame_count += 1
                        
                except (av.AVError, RuntimeError) as e:
                    logger.error(f"PyAV decoder error processing video {video_path} at frame {frame_count}: {e}")
                    logger.error(f"Possible H.264 NAL unit corruption detected, stopping decoding")
                
                finally:
                    if container:
                        container.close()
        
        except Exception as e:
            logger.error(f"PyAV failed to process video {video_path}: {e}")
            if container:
                container.close()
            return
    else:
        vr = None
        try:
            import decord  # type: ignore
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        except ImportError:
            logger.error(f"decord is not available for video processing. Install with: pip install vlm-engine[decord]")
            return
        except RuntimeError as e:
            logger.error(f"Decord failed to open video {video_path}: {e}")
            return
            
        fps: float = vr.get_avg_fps()
        if fps == 0:
            logger.warning(f"Video {video_path} has FPS of 0. Cannot process.")
            if vr: del vr
            return

        frames_to_skip: int = custom_round(fps * frame_interval_sec) 
        if frames_to_skip < 1: frames_to_skip = 1

        if process_for_vlm:
            MAX_DECORD_FRAMES = 10000  # Safety limit for decord as well
            processed_frames = 0
            
            for i in range(0, len(vr), frames_to_skip):
                if processed_frames >= MAX_DECORD_FRAMES:
                    logger.warning(f"Hit decord frame limit {MAX_DECORD_FRAMES}, possible issues")
                    break
                    
                try:
                    frame_cpu = vr[i] 
                except RuntimeError as e_read_frame:
                    logger.warning(f"Could not read frame {i} from {video_path}: {e_read_frame}")
                    processed_frames += 1
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
                processed_frames += 1
        else:
            logger.warning("Standard processing path no longer supported - use VLM processing")
            return
                
        if vr: del vr

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
