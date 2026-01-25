"""
Efficiently extracts specific frames from video files with parallel processing and caching
"""

from .action_range import ActionRange
from .adaptive_midpoint_collector import AdaptiveMidpointCollector
import asyncio
import logging
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
import numpy as np
from .preprocessing import crop_black_bars_lr, is_macos_arm, _pyav_semaphore

if is_macos_arm:
    try:
        import av
    except ImportError:
        av = None
else:
    try:
        import decord
        # Ensure bridge is set to torch for proper operation
        decord.bridge.set_bridge('torch')
    except ImportError:
        decord = None

class VideoFrameExtractor:
    """Efficiently extracts specific frames from video files with parallel processing and caching"""
    
    def __init__(self, use_half_precision: bool = True, max_workers: int = 6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_half_precision = use_half_precision
        self.max_workers = max_workers
        self.logger = logging.getLogger("logger")
        self.frame_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self.cache_size_limit = 20  # Reduced for better memory management on large videos
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def extract_frame(self, video_path: str, frame_idx: int) -> Optional[torch.Tensor]:
        """Extract a specific frame from video with caching"""
        cache_key = (video_path, frame_idx)
        
        # Check cache first
        if cache_key in self.frame_cache:
            self.logger.debug(f"Cache hit for frame {frame_idx}")
            return self.frame_cache[cache_key]
        
        try:
            if is_macos_arm:
                frame_tensor = self._extract_frame_pyav(video_path, frame_idx)
            else:
                frame_tensor = self._extract_frame_decord(video_path, frame_idx)
            
            # Cache the frame if extraction was successful
            if frame_tensor is not None:
                self._cache_frame(cache_key, frame_tensor)
            
            return frame_tensor
        except Exception as e:
            self.logger.error(f"Failed to extract frame {frame_idx} from {video_path}: {e}")
            return None
    
    async def extract_frames_parallel(self, video_path: str, frame_indices: List[int]) -> Dict[int, Optional[torch.Tensor]]:
        """Extract multiple frames in parallel"""
        results = {}
        
        # Check cache for existing frames
        uncached_indices = []
        for frame_idx in frame_indices:
            cache_key = (video_path, frame_idx)
            if cache_key in self.frame_cache:
                results[frame_idx] = self.frame_cache[cache_key]
                self.logger.debug(f"Cache hit for frame {frame_idx}")
            else:
                uncached_indices.append(frame_idx)
        
        if not uncached_indices:
            return results
        
        # Extract uncached frames in parallel
        loop = asyncio.get_event_loop()
        
        async def extract_single_frame(frame_idx: int) -> Tuple[int, Optional[torch.Tensor]]:
            try:
                frame_tensor = await loop.run_in_executor(
                    self.executor, 
                    self._extract_frame_sync, 
                    video_path, 
                    frame_idx
                )
                if frame_tensor is not None:
                    cache_key = (video_path, frame_idx)
                    self._cache_frame(cache_key, frame_tensor)
                return frame_idx, frame_tensor
            except Exception as e:
                self.logger.error(f"Failed to extract frame {frame_idx}: {e}")
                return frame_idx, None
        
        # Execute all extractions in parallel
        extraction_tasks = [extract_single_frame(idx) for idx in uncached_indices]
        extraction_results = await asyncio.gather(*extraction_tasks)
        
        # Combine results
        for frame_idx, frame_tensor in extraction_results:
            results[frame_idx] = frame_tensor
        
        return results
    
    def _extract_frame_sync(self, video_path: str, frame_idx: int) -> Optional[torch.Tensor]:
        """Synchronous frame extraction for use in thread pool"""
        try:
            if is_macos_arm:
                return self._extract_frame_pyav(video_path, frame_idx)
            else:
                return self._extract_frame_decord(video_path, frame_idx)
        except Exception as e:
            self.logger.error(f"Failed to extract frame {frame_idx} from {video_path}: {e}")
            return None
    
    def _cache_frame(self, cache_key: Tuple[str, int], frame_tensor: torch.Tensor) -> None:
        """Cache a frame with size limit management"""
        if len(self.frame_cache) >= self.cache_size_limit:
            # Remove oldest entry (simple FIFO eviction)
            oldest_key = next(iter(self.frame_cache))
            del self.frame_cache[oldest_key]
            self.logger.debug(f"Evicted cached frame {oldest_key[1]} from {oldest_key[0]}")
            
            # NEW: Force GC after eviction
            import gc
            gc.collect()
        
        self.frame_cache[cache_key] = frame_tensor.clone()  # Clone to avoid reference issues
        self.logger.debug(f"Cached frame {cache_key[1]} from {cache_key[0]}")
    
    def clear_cache(self) -> None:
        """Clear the frame cache"""
        self.frame_cache.clear()
        self.logger.debug("Frame cache cleared")
    
    def __del__(self):
        """Cleanup thread pool executor"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
    
    def _extract_frame_decord(self, video_path: str, frame_idx: int) -> Optional[torch.Tensor]:
        """Extract frame using decord"""
        if decord is None:
            self.logger.error("decord is not available for video processing. Install with: pip install vlm-engine[decord]")
            return None
            
        try:
            # Add timeout protection for decord initialization
            import threading
            import queue
            
            result_queue = queue.Queue()
            error_queue = queue.Queue()
            
            def extract_frame_thread():
                try:
                    # Ensure bridge is set to torch before instantiating VideoReader
                    try:
                        decord.bridge.set_bridge('torch')
                    except Exception:
                        pass
                    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))  # No readahead for 0.6.0
                    self.logger.debug(f'Created VideoReader for {video_path}')
                    
                    # Safety check for frame index
                    if frame_idx >= len(vr):
                        self.logger.warning(f"Frame index {frame_idx} exceeds video length {len(vr)}")
                        del vr
                        result_queue.put(None)
                        return
                    
                    frame_cpu = vr[frame_idx]
                    if not isinstance(frame_cpu, torch.Tensor):
                        frame_cpu = torch.from_numpy(frame_cpu.asnumpy())
                    
                    frame_cpu = crop_black_bars_lr(frame_cpu)
                    frame = frame_cpu.to(self.device)
                    
                    if not torch.is_floating_point(frame):
                        frame = frame.float() / 255.0
                    if self.use_half_precision:
                        frame = frame.half()
                    
                    del vr
                    self.logger.debug(f'Released VideoReader after extracting frame {frame_idx}')
                    result_queue.put(frame)
                except Exception as e:
                    error_queue.put(e)
            
            # Run decord extraction in a thread with timeout
            thread = threading.Thread(target=extract_frame_thread, daemon=True)
            thread.start()
            thread.join(timeout=30.0)  # 30 second timeout
            
            if thread.is_alive():
                self.logger.error(f"Decord frame extraction timed out after 30s for frame {frame_idx}")
                return None
            
            if not error_queue.empty():
                error = error_queue.get()
                self.logger.error(f"Decord frame extraction failed: {error}")
                return None
            
            if not result_queue.empty():
                return result_queue.get()
            
            return None
        except Exception as e:
            self.logger.error(f"Decord frame extraction failed: {e}")
            return None
    
    def _extract_frame_pyav(self, video_path: str, frame_idx: int) -> Optional[torch.Tensor]:
        """Extract frame using PyAV with resource management and validation"""
        if av is None:
            self.logger.error("PyAV is not available for frame extraction")
            return None
        
        try:
            if frame_idx < 0:
                self.logger.warning(f"Frame index {frame_idx} must be non-negative.")
                return None

            # Use semaphore to limit concurrent PyAV operations
            with _pyav_semaphore:
                with av.open(video_path, stream_options={'err_detect': 'ignore_err'}) as container:
                    try:
                        stream = container.streams.video[0]
                        fps = float(stream.average_rate)
                        total_frames = stream.frames or 0
                        
                        # Skip initial frames not yet present
                        initial_padding = stream.start_time if hasattr(stream, "start_time") and stream.start_time else 0.0
                        seek_frame = max(0, frame_idx - initial_padding * fps)
                        if seek_frame < 0:
                            self.logger.warning(f"Calculated seek_frame {seek_frame} is invalid after adjusting for initial padding")
                            return None
                        
                        # Seek to approximate time
                        timestamp = int(seek_frame / fps * av.time_base)
                        container.seek(timestamp, stream=stream)
                        
                        current_frame = 0
                        MAX_DECODE_ITERATIONS = 1000  # Additional safety limit for seeking
                        try:
                            decode_context = container.decode(stream)
                            for frame in decode_context:
                                if current_frame >= MAX_DECODE_ITERATIONS:
                                    self.logger.warning(f"Hit decode iteration limit {MAX_DECODE_ITERATIONS} while seeking frame {seek_frame}, possible corruption")
                                    break
                                
                                if current_frame == seek_frame:
                                    frame_np = frame.to_ndarray(format='rgb24')
                                    frame_tensor = torch.from_numpy(frame_np).to(self.device)
                                    frame_tensor = crop_black_bars_lr(frame_tensor)
                                    
                                    if not torch.is_floating_point(frame_tensor):
                                        frame_tensor = frame_tensor.float()
                                    
                                    if self.use_half_precision:
                                        frame_tensor = frame_tensor.half()
                                    
                                    return frame_tensor
                                current_frame += 1
                                
                                if current_frame > seek_frame + 50:
                                    # Safety threshold to avoid excessive decoding
                                    self.logger.warning(f"Exceeded frame seek threshold seeking {seek_frame}")
                                    break
                        except (av.AVError, RuntimeError) as e:
                            self.logger.error(f"PyAV decoder error while seeking frame {seek_frame}: {e}")
                            self.logger.error(f"Possible H.264 NAL unit corruption detected, stopping decoding at frame {current_frame}")
                            return None
                    except Exception as e:
                        self.logger.error(f"PyAV frame extraction error: {e}")
                        return None

                    self.logger.warning(f"Frame index {frame_idx} ({seek_frame} after seek) not found in video")
                    return None
        except Exception as e:
            self.logger.error(f"Failed to open video file for PyAV extraction: {e}")
            return None
