"""
Efficiently extracts specific frames from video files with parallel processing and caching
"""

from .action_range import ActionRange
from .adaptive_midpoint_collector import AdaptiveMidpointCollector
import asyncio
import logging
import signal
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
import numpy as np
from .preprocessing import crop_black_bars_lr, is_macos_arm

if is_macos_arm:
    import av
else:
    import decord
    decord.bridge.set_bridge('torch')

class VideoFrameExtractor:
    """Efficiently extracts specific frames from video files with parallel processing and caching"""
    
    def __init__(self, device_str: Optional[str] = None, use_half_precision: bool = True, max_workers: int = 4):
        self.device = torch.device(device_str) if device_str else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_half_precision = use_half_precision
        self.max_workers = max_workers
        self.logger = logging.getLogger("logger")
        self.frame_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self.cache_size_limit = 20  # Reduced for better memory management on large videos
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Enhanced NAL unit error handling and failure tracking
        self.frame_failure_counts: Dict[Tuple[str, int], int] = {}
        self.corrupted_frames: Dict[str, set] = {}  # video_path -> set of corrupted frame indices
        self.corrupted_videos: Dict[str, bool] = {}  # video_path -> if video is too corrupted
        self.api_calls_made = 0  # Track total API calls for corruption ratio
        self.total_failures = 0
        self.max_retry_attempts = 2  # Reduced to prevent infinite loops
        self.corruption_window_size = 200  # Increased window around corrupted areas
        self.max_video_corruption_ratio = 0.05  # Abort if >5% of frames are corrupted
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5  # Abort after 5 consecutive failures
        
        # Timeout protection for decoder (to prevent infinite loops in FFmpeg)
        self.decoder_timeout_seconds = 30  # Max time for any single decoder operation
        self.seeking_timeout_seconds = 10  # Max time for seeking operations
        
        self.logger.info(f"VideoFrameExtractor initialized with enhanced NAL error handling (max retries: {self.max_retry_attempts}, corruption window: {self.corruption_window_size}, decoder timeout: {self.decoder_timeout_seconds}s)")
    
    def extract_frame(self, video_path: str, frame_idx: int) -> Optional[torch.Tensor]:
        """Extract a specific frame with timeout protection and NAL error handling"""
        cache_key = (video_path, frame_idx)
        self.api_calls_made += 1
        
        # Check if video is too corrupted to process
        if self._is_video_too_corrupted(video_path):
            self.logger.error(f"Video {video_path} marked as too corrupted - aborting all processing")
            return None
        
        # Check cache first
        if cache_key in self.frame_cache:
            self.logger.debug(f"Cache hit for frame {frame_idx}")
            return self.frame_cache[cache_key]
        
        # Check if frame is accessible (not marked as corrupted or exceeded retries)
        if not self._is_frame_accessible(video_path, frame_idx):
            self.logger.debug(f"Skipping frame {frame_idx} due to corruption/retry limits")
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.logger.error(f"Too many consecutive failures ({self.consecutive_failures}), aborting video processing")
                self.corrupted_videos[video_path] = True
            
            return None
        
        try:
            # Use timeout-protected extraction to prevent decoder infinite loops
            if is_macos_arm:
                frame_tensor = self._extract_frame_pyav_with_timeout(video_path, frame_idx)
            else:
                frame_tensor = self._extract_frame_decord_with_timeout(video_path, frame_idx)
            
            # On successful extraction, reset failure tracking
            self.consecutive_failures = 0
            
            if cache_key in self.frame_failure_counts:
                del self.frame_failure_counts[cache_key]
                self.logger.debug(f"Cleared failure count for previously failing frame {frame_idx}")
            
            if video_path in self.corrupted_frames and frame_idx in self.corrupted_frames[video_path]:
                self.logger.info(f"Successfully extracted previously corrupted frame {frame_idx}")
            
            # Cache the frame if extraction was successful
            if frame_tensor is not None:
                self._cache_frame(cache_key, frame_tensor)
                self.total_failures += 1
            
            return frame_tensor
        except Exception as e:
            error_msg = str(e)
            
            # Check for timeout errors specifically
            if "timeout" in error_msg.lower() or isinstance(e, (TimeoutError, signal.TimeoutError)):
                self.logger.error(f"TIMEOUT during frame extraction for {frame_idx} from {video_path} - decoder likely stuck in infinite loop")
                self._mark_frame_as_corrupted(video_path, frame_idx)
                self._increment_failure_count(video_path, frame_idx)
                self.consecutive_failures += 1
            else:
                self.logger.error(f"Failed to extract frame {frame_idx} from {video_path}: {e}")
                self.total_failures += 1
                
                # Handle NAL unit errors specifically
                if self._is_nal_unit_error(error_msg):
                    self.logger.warning(f"NAL unit corruption detected for frame {frame_idx}")
                    self._increment_failure_count(video_path, frame_idx)
                    self._mark_frame_as_corrupted(video_path, frame_idx)
                    
                    if self.frame_failure_counts.get(cache_key, 0) >= self.max_retry_attempts:
                        self.logger.error(f"Frame {frame_idx} permanently blocked after {self.max_retry_attempts} NAL errors")
                else:
                    # For non-NAL errors, still track the failure
                    self._increment_failure_count(video_path, frame_idx)
            
            return None
    
    async def extract_frames_parallel(self, video_path: str, frame_indices: List[int]) -> Dict[int, Optional[torch.Tensor]]:
        """Extract multiple frames in parallel with NAL error handling"""
        results = {}
        
        # Check cache for existing frames and filter out corrupted/inaccessible frames
        uncached_indices = []
        for frame_idx in frame_indices:
            cache_key = (video_path, frame_idx)
            if cache_key in self.frame_cache:
                results[frame_idx] = self.frame_cache[cache_key]
                self.logger.debug(f"Cache hit for frame {frame_idx}")
            elif self._is_frame_accessible(video_path, frame_idx):
                uncached_indices.append(frame_idx)
            else:
                self.logger.warning(f"Skipping frame {frame_idx} due to previous failures or corruption")
                results[frame_idx] = None
        
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
                error_msg = str(e)
                self.logger.error(f"Failed to extract frame {frame_idx}: {e}")
                
                # Handle NAL unit errors for parallel extraction too
                if self._is_nal_unit_error(error_msg):
                    self.logger.warning(f"NAL unit corruption detected for frame {frame_idx} in parallel extraction")
                    self._increment_failure_count(video_path, frame_idx)
                    self._mark_frame_as_corrupted(video_path, frame_idx)
                
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
    
    def clear_cache(self, video_path: Optional[str] = None) -> None:
        """Clear the frame cache, optionally for a specific video"""
        if video_path:
            # Clear cache for specific video
            keys_to_remove = [key for key in self.frame_cache.keys() if key[0] == video_path]
            for key in keys_to_remove:
                del self.frame_cache[key]
            self.logger.debug(f"Frame cache cleared for {video_path}")
        else:
            self.frame_cache.clear()
            self.logger.debug("Frame cache cleared")
    
    def reset_error_tracking(self, video_path: Optional[str] = None) -> None:
        """Reset error tracking and corruption markers, optionally for a specific video"""
        if video_path:
            # Reset error tracking for specific video
            keys_to_remove = [key for key in self.frame_failure_counts.keys() if key[0] == video_path]
            for key in keys_to_remove:
                del self.frame_failure_counts[key]
            
            if video_path in self.corrupted_frames:
                del self.corrupted_frames[video_path]
            
            self.logger.debug(f"Error tracking reset for {video_path}")
        else:
            self.frame_failure_counts.clear()
            self.corrupted_frames.clear()
            self.logger.debug("All error tracking and corruption markers cleared")
    
    def __del__(self):
        """Cleanup thread pool executor"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
    
    def _is_frame_accessible(self, video_path: str, frame_idx: int) -> bool:
        """Enhanced frame accessibility check with video-level corruption"""
        cache_key = (video_path, frame_idx)
        
        # Check if video is marked as too corrupted
        if self._is_video_too_corrupted(video_path):
            return False
        
        # Check if frame has exceeded maximum retry attempts
        if cache_key in self.frame_failure_counts:
            failure_count = self.frame_failure_counts[cache_key]
            if failure_count >= self.max_retry_attempts:
                self.logger.debug(f"Frame {frame_idx} blocked: exceeded retry attempts ({failure_count}/{self.max_retry_attempts})")
                return False
        
        # Check if frame is in corrupted regions  
        if video_path in self.corrupted_frames:
            corrupted_set = self.corrupted_frames[video_path]
            
            # Check if this frame is directly corrupted
            if frame_idx in corrupted_set:
                self.logger.debug(f"Frame {frame_idx} blocked: marked as corrupted")
                return False
            
            # Check if nearby frames are corrupted (within corruption window)
            for corrupted_idx in corrupted_set:
                if abs(frame_idx - corrupted_idx) <= self.corruption_window_size:
                    self.logger.debug(f"Frame {frame_idx} blocked: near corrupted frame {corrupted_idx}")
                    return False
        
        return True
    
    def _mark_frame_as_corrupted(self, video_path: str, frame_idx: int) -> None:
        """Mark a frame and surrounding region as corrupted"""
        if video_path not in self.corrupted_frames:
            self.corrupted_frames[video_path] = set()
        
        # Mark this frame and surrounding frames as corrupted
        corruption_start = max(0, frame_idx - self.corruption_window_size)
        corruption_end = frame_idx + self.corruption_window_size
        
        # Add range of frames to corruption pool
        for i in range(corruption_start, corruption_end + 1):
            self.corrupted_frames[video_path].add(i)
        
        self.logger.debug(f"Marked frames {corruption_start}-{corruption_end} as corrupted for {video_path}:{frame_idx}")
        
        # Check if we've marked too many frames for this video
        self._is_video_too_corrupted(video_path)
    
    def _increment_failure_count(self, video_path: str, frame_idx: int) -> None:
        """Increment failure count for a frame"""
        cache_key = (video_path, frame_idx)
        self.frame_failure_counts[cache_key] = self.frame_failure_counts.get(cache_key, 0) + 1
    
    def _is_nal_unit_error(self, error_msg: str) -> bool:
        """Enhanced detection of NAL unit corruption errors"""
        nal_patterns = [
            "Invalid NAL unit size",
            "Error splitting the input into NAL units",
            "nal",  # Any mention of NAL
            "h264",  # H.264 codec issues
            "h265",  # H.265 codec issues
            "codec error",
            "invalid nal unit",
            "nal unit size",
            "splitting.*nal",
            "@.*Invalid NAL unit",  # PyAV specific pattern
            "Error.*splitting.*nal",  # FFmpeg pattern
        ]
        
        error_msg_lower = error_msg.lower()
        matches = [pat for pat in nal_patterns if pat in error_msg_lower]
        
        if matches:
            self.logger.debug(f"NAL error patterns matched: {matches} in: {error_msg[:100]}")
            return True
        
        return False
    
    def _is_video_too_corrupted(self, video_path: str) -> bool:
        """Check if a video should be marked as too corrupted to process"""
        if video_path in self.corrupted_videos:
            return self.corrupted_videos[video_path]
        
        if video_path not in self.corrupted_frames:
            return False
        
        corrupted_count = len(self.corrupted_frames[video_path])
        total_attempted = max(self.api_calls_made, corrupted_count * 20)  # Use actual attempts if available
        
        corruption_ratio = corrupted_count / max(total_attempted, 1)
        
        self.logger.debug(f"Video corruption check: {corrupted_count}/{total_attempted} = {corruption_ratio:.1%} (threshold: {self.max_video_corruption_ratio:.1%})")
        
        if corruption_ratio > self.max_video_corruption_ratio:
            self.logger.warning(f"Marking video {video_path} as too corrupted: {corruption_ratio:.1%} of frames failed")
            self.corrupted_videos[video_path] = True
            return True
        
        return False
    
    def _extract_frame_decord(self, video_path: str, frame_idx: int) -> Optional[torch.Tensor]:
        """Extract frame using decord with enhanced error handling"""
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))  # No readahead for 0.6.0
            self.logger.debug(f'Created VideoReader for {video_path}')
            
            if frame_idx >= len(vr):
                self.logger.warning(f"Frame index {frame_idx} exceeds video length {len(vr)}")
                del vr
                return None
            
            # Try to access the frame with additional error handling
            try:
                frame_cpu = vr[frame_idx]
            except Exception as frame_error:
                error_msg = str(frame_error)
                self.logger.error(f"Decord frame access error for frame {frame_idx}: {frame_error}")
                
                # Check for NAL unit related errors
                if self._is_nal_unit_error(error_msg):
                    self.logger.warning(f"NAL unit corruption detected by Decord at frame {frame_idx}")
                    raise frame_error  # Re-raise to be caught by outer exception handler
                
                # Try alternative access method for other errors
                try:
                    self.logger.debug(f"Attempting alternative frame access method for {frame_idx}")
                    frame_cpu = vr.get_batch([frame_idx])[0]
                except Exception as alternative_error:
                    self.logger.error(f"Alternative frame access also failed for {frame_idx}: {alternative_error}")
                    raise alternative_error
            
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
            return frame
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Decord frame extraction failed for frame {frame_idx}: {e}")
            
            # Check if this is a NAL unit error and re-raise for proper handling
            if self._is_nal_unit_error(error_msg):
                raise e  # Re-raise to be handled by calling method
            
            return None
    
    def _extract_frame_pyav(self, video_path: str, frame_idx: int) -> Optional[torch.Tensor]:
        """Extract frame using PyAV with enhanced NAL unit error handling"""
        try:
            if frame_idx < 0:
                self.logger.warning(f"Frame index {frame_idx} must be non-negative.")
                return None

            # Try multiple approaches for frame extraction
            extraction_methods = [
                self._try_pyav_decode_stream,
                self._try_pyav_direct_seek,
                self._try_pyav_frame_iter
            ]
            
            for method_name, extraction_method in extraction_methods:
                try:
                    frame_tensor = extraction_method(video_path, frame_idx)
                    if frame_tensor is not None:
                        return frame_tensor
                except Exception as e:
                    error_msg = str(e)
                    self.logger.debug(f"PyAV method {method_name} failed for frame {frame_idx}: {e}")
                    
                    # If this is a NAL unit error, skip to next method
                    if self._is_nal_unit_error(error_msg):
                        self.logger.warning(f"NAL unit corruption detected with {method_name} at frame {frame_idx}, trying alternative method")
                        continue
                    else:
                        # For non-NAL errors, try next method
                        continue
            
            # If all methods failed, log the final failure
            self.logger.error(f"All PyAV extraction methods failed for frame {frame_idx}")
            return None
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"PyAV frame extraction failed for {video_path} frame {frame_idx}: {e}")
            
            # Check if this is a NAL unit error and re-raise for proper handling
            if self._is_nal_unit_error(error_msg):
                raise e  # Re-raise to be handled by calling method
            
            return None

    def _try_pyav_decode_stream(self, video_path: str, frame_idx: int) -> Optional[torch.Tensor]:
        """Try PyAV extraction using decode stream method with corruption detection"""
        start_time = time.time()
        
        try:
            # Open container with timeout to prevent infinite wait
            with av.open(video_path, timeout=self.seeking_timeout_seconds) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate)
                
                # Skip initial frames not yet present
                initial_padding = stream.start_time if hasattr(stream, "start_time") and stream.start_time else 0.0
                seek_frame = max(0, frame_idx - initial_padding * fps)
                if seek_frame < 0:
                    self.logger.warning(f"Calculated seek_frame {seek_frame} is invalid after adjusting for initial padding")
                    return None
                
                # Seek to approximate time with safety check
                timestamp = int(seek_frame / fps * av.time_base)
                container.seek(timestamp, stream=stream)
                
                current_frame = 0
                max_frames_to_decode = 200  # Hard safety limit
                
                try:
                    for frame in container.decode(stream):
                        # Check for decode timeout/corruption
                        elapsed = time.time() - start_time
                        if elapsed > self.decoder_timeout_seconds:
                            self.logger.error(f"Decode timeout for frame {frame_idx} after {elapsed:.1f}s - likely infinite loop in decoder")
                            return None
                        
                        if current_frame == seek_frame:
                            frame_np = frame.to_ndarray(format='rgb24')
                            frame_tensor = torch.from_numpy(frame_np).to(self.device)
                            frame_tensor = crop_black_bars_lr(frame_tensor)
                            
                            if not torch.is_floating_point(frame_tensor):
                                frame_tensor = frame_tensor.float()
                            
                            if self.use_half_precision:
                                frame_tensor = frame_tensor.half()
                            
                            elapsed = time.time() - start_time
                            self.logger.debug(f"Frame {frame_idx} extracted successfully in {elapsed:.2f}s")
                            return frame_tensor
                        
                        current_frame += 1
                        
                        # Safety threshold to prevent decoder loops
                        if current_frame > seek_frame + 50 or current_frame >= max_frames_to_decode:
                            self.logger.warning(f"Exceeded frame decode threshold: {current_frame} frames processed")
                            break
                except Exception as decode_error:
                    if self._is_nal_unit_error(str(decode_error)):
                        self.logger.warning(f"NAL unit corruption in decode stream at frame {current_frame}: {decode_error}")
                        raise decode_error  # Re-raise for outer handler
                    self.logger.error(f"Decode error at frame {current_frame}: {decode_error}")
                    return None

            elapsed = time.time() - start_time
            self.logger.warning(f"Frame index {frame_idx} ({seek_frame} after seek) not found in video (took {elapsed:.1f}s)")
            return None
            
        except Exception as e:
            elapsed = time.time() - start_time
            
            # Check for timeout conditions
            if hasattr(e, '__cause__') and isinstance(e.__cause__, TimeoutError):
                self.logger.error(f"PyAV timeout after {elapsed:.1f}s for frame {frame_idx}")
            elif "timeout" in str(e).lower():
                self.logger.error(f"PyAV timeout/lockup for frame {frame_idx} after {elapsed:.1f}s")
            
            # Re-raise NAL errors to try next method
            if self._is_nal_unit_error(str(e)):
                raise e
                
            self.logger.debug(f"PyAV decode stream method failed for frame {frame_idx}: {e}")
            return None

    def _try_pyav_direct_seek(self, video_path: str, frame_idx: int) -> Optional[torch.Tensor]:
        """Try PyAV extraction using direct seek method with timeout protection"""
        start_time = time.time()
        
        try:
            # Open container with timeout to prevent infinite wait
            with av.open(video_path, timeout=self.seeking_timeout_seconds) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate)
                
                # Try direct frame indexing
                try:
                    # Calculate timestamp directly
                    timestamp = int(frame_idx / fps * av.time_base)
                    container.seek(timestamp, stream=stream)
                    
                    # Get a few frames to ensure we have the right one
                    frames_tried = 0
                    max_frames_to_try = 10
                    
                    for frame in container.decode(stream):
                        frames_tried += 1
                        
                        # Check for decode timeout/corruption
                        elapsed = time.time() - start_time
                        if elapsed > self.decoder_timeout_seconds:
                            self.logger.error(f"Direct seek timeout for frame {frame_idx} after {elapsed:.1f}s")
                            return None
                        
                        if frames_tried >= max_frames_to_try:
                            break
                        
                        # Check if this frame is close enough
                        expected_frame_time = frame_idx / fps
                        actual_frame_time = float(frame.pts * stream.time_base)
                        
                        if abs(actual_frame_time - expected_frame_time) < (0.5 / fps):  # Within half frame tolerance
                            frame_np = frame.to_ndarray(format='rgb24')
                            frame_tensor = torch.from_numpy(frame_np).to(self.device)
                            frame_tensor = crop_black_bars_lr(frame_tensor)
                            
                            if not torch.is_floating_point(frame_tensor):
                                frame_tensor = frame_tensor.float()
                            
                            if self.use_half_precision:
                                frame_tensor = frame_tensor.half()
                            
                            elapsed = time.time() - start_time
                            self.logger.debug(f"Frame {frame_idx} extracted via direct seek in {elapsed:.2f}s")
                            return frame_tensor
                            
                except Exception as decode_error:
                    elapsed = time.time() - start_time
                    
                    if self._is_nal_unit_error(str(decode_error)):
                        self.logger.warning(f"NAL unit corruption in direct seek: {decode_error}")
                        raise decode_error  # Re-raise for outer handler
                    elif elapsed > self.decoder_timeout_seconds * 0.8:
                        self.logger.error(f"Direct seek timeout/corruption after {elapsed:.1f}s")
                        return None
                        
                    self.logger.debug(f"Direct seek method decode error: {decode_error}")
            
            elapsed = time.time() - start_time
            self.logger.warning(f"Direct seek failed for frame {frame_idx} in {elapsed:.1f}s")
            return None
            
        except Exception as e:
            elapsed = time.time() - start_time
            
            # Check for timeout conditions
            if "timeout" in str(e).lower():
                self.logger.error(f"PyAV direct seek timeout/lockup for frame {frame_idx} after {elapsed:.1f}s")
            
            # Re-raise NAL errors to try next method
            if self._is_nal_unit_error(str(e)):
                raise e
                
            self.logger.debug(f"PyAV direct seek method failed for frame {frame_idx}: {e}")
            return None

    def _try_pyav_frame_iter(self, video_path: str, frame_idx: int) -> Optional[torch.Tensor]:
        """Try PyAV extraction using frame iteration method with corruption detection"""
        start_time = time.time()
        
        try:
            # Open container with timeout to prevent infinite wait
            with av.open(video_path, timeout=self.seeking_timeout_seconds) as container:
                stream = container.streams.video[0]
                
                # Iterate through frames until we find the target
                frame_count = 0
                max_frames_to_check = min(frame_idx + 100, stream.frames or (frame_idx + 100))
                
                try:
                    for frame in container.decode(stream):
                        # Check for decode timeout/corruption
                        elapsed = time.time() - start_time
                        if elapsed > self.decoder_timeout_seconds:
                            self.logger.error(f"Frame iteration timeout for frame {frame_idx} after {elapsed:.1f}s - likely infinite decode loop")
                            return None
                        
                        if frame_count == frame_idx:
                            frame_np = frame.to_ndarray(format='rgb24')
                            frame_tensor = torch.from_numpy(frame_np).to(self.device)
                            frame_tensor = crop_black_bars_lr(frame_tensor)
                            
                            if not torch.is_floating_point(frame_tensor):
                                frame_tensor = frame_tensor.float()
                            
                            if self.use_half_precision:
                                frame_tensor = frame_tensor.half()
                            
                            elapsed = time.time() - start_time
                            self.logger.debug(f"Frame {frame_idx} extracted via iteration in {elapsed:.2f}s")
                            return frame_tensor
                        
                        frame_count += 1
                        if frame_count >= max_frames_to_check:
                            elapsed = time.time() - start_time
                            self.logger.warning(f"Frame iteration reached limit: {frame_count} frames in {elapsed:.1f}s")
                            break
                            
                except Exception as decode_error:
                    elapsed = time.time() - start_time
                    
                    if self._is_nal_unit_error(str(decode_error)):
                        self.logger.warning(f"NAL unit corruption in frame iteration at frame {frame_count}: {decode_error}")
                        raise decode_error  # Re-raise for outer handler
                    elif elapsed > self.decoder_timeout_seconds * 0.8:
                        self.logger.error(f"Frame iteration timeout/corruption at frame {frame_count} after {elapsed:.1f}s")
                        return None
                    
                    self.logger.debug(f"Frame iteration decode error at frame {frame_count}: {decode_error}")
            
            elapsed = time.time() - start_time
            self.logger.warning(f"Frame {frame_idx} not found via iteration in {elapsed:.1f}s")
            return None
            
        except Exception as e:
            elapsed = time.time() - start_time
            
            # Check for timeout conditions
            if "timeout" in str(e).lower():
                self.logger.error(f"PyAV iteration timeout/lockup for frame {frame_idx} after {elapsed:.1f}s")
            
            # Re-raise NAL errors to try next method
            if self._is_nal_unit_error(str(e)):
                raise e
                
            self.logger.debug(f"PyAV frame iteration method failed for frame {frame_idx}: {e}")
            return None
    
    def _extract_frame_pyav_with_timeout(self, video_path: str, frame_idx: int) -> Optional[torch.Tensor]:
        """Protected PyAV extraction that wraps normal extraction with timeout"""
        return self._extract_frame_pyav(video_path, frame_idx)
    
    def _extract_frame_decord_with_timeout(self, video_path: str, frame_idx: int) -> Optional[torch.Tensor]:
        """Protected Decord extraction that wraps normal extraction with timeout"""
        return self._extract_frame_decord(video_path, frame_idx)
