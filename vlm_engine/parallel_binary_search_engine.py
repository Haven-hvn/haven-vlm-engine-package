"""
Main engine implementing sliding window + binary search for action detection.
Replaces linear scan with sliding window technique for better action segmentation.
"""

import asyncio
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, cast
import torch
import numpy as np
from PIL import Image

from .action_range import ActionRange
from .adaptive_midpoint_collector import AdaptiveMidpointCollector
from .action_boundary_detector import ActionBoundaryDetector
from .video_frame_extractor import VideoFrameExtractor
from .preprocessing import get_video_duration_decord, is_macos_arm

try:
    import decord  # type: ignore
except ImportError:
    decord = None
try:
    import av  # type: ignore
except ImportError:
    av = None

class ParallelBinarySearchEngine:
    """
    Main engine implementing sliding window + binary search for action detection.
    Uses sliding windows for detection, then binary search within windows for precision.
    """
    
    def __init__(
        self, 
        action_tags: Optional[List[str]] = None,
        threshold: float = 0.5,
        device_str: Optional[str] = None,
        use_half_precision: bool = True
    ):
        self.action_tags = action_tags or []
        self.threshold = threshold
        self.logger = logging.getLogger("logger")
        
        # Core components
        self.midpoint_collector = AdaptiveMidpointCollector()
        self.boundary_detector = ActionBoundaryDetector(threshold)
        self.frame_extractor = VideoFrameExtractor(device_str, use_half_precision)
        
        # Search state
        self.action_ranges: List[ActionRange] = []
        self.candidate_windows: List[Dict[str, Any]] = []  # Windows with detected actions
        self.total_frames = 0
        self.api_calls_made = 0
        
        # VLM analysis result caching
        self.vlm_cache: Dict[Tuple[str, int], Dict[str, float]] = {}
        self.vlm_cache_size_limit = 200
        
        self.logger.info(f"ParallelBinarySearchEngine initialized for {len(self.action_tags)} actions")
    
    def has_unresolved_actions(self) -> bool:
        """Check if there are still actions being searched"""
        return any(not action_range.is_resolved() for action_range in self.action_ranges)
    
    def _has_actions_within_depth_limit(self) -> bool:
        """Check if any actions are still within their depth limits"""
        return any(
            not action_range.is_resolved() and not action_range.has_reached_max_depth()
            for action_range in self.action_ranges
        )

    def _calculate_adaptive_window(self, total_frames: int, fps: float) -> Tuple[float, float]:
        """Calculate adaptive window size and overlap based on video duration"""
        duration = total_frames / fps  # Video duration in seconds
        
        # Duration-based scaling from 5s (minimum) to 15s (maximum)
        min_window_size = 5.0
        max_window_size = 40.0
        
        # Calculate window size using logarithmic scaling
        window_size = math.ceil(math.log2(duration)) + 4
        window_size = max(min_window_size, min(window_size, max_window_size))
        
        # Adaptive overlap ratio between 20-40% based on window size
        overlap_ratio = 0.4 - (0.2 * (window_size - min_window_size) / (max_window_size - min_window_size))
        window_overlap = window_size * overlap_ratio
        
        self.logger.info(
            f"Adaptive windows: duration={duration:.1f}s, size={window_size:.1f}s, overlap={window_overlap:.1f}s"
        )
        
        return window_size, window_overlap

    def _create_sliding_windows(self, total_frames: int, fps: float) -> List[Dict[str, int]]:
        """Create sliding windows based on adaptive duration scaling"""
        # Calculate adaptive window parameters
        window_size_sec, window_overlap_sec = self._calculate_adaptive_window(total_frames, fps)
        
        # Update instance variables to reflect adaptive values
        self.window_size_sec = window_size_sec
        self.window_overlap_sec = window_overlap_sec
        min_window_frames = 10
        
        window_size_frames = max(
            min_window_frames, 
            int(self.window_size_sec * fps)
        )
        window_step = max(
            1, 
            int((self.window_size_sec - self.window_overlap_sec) * fps)
        )
        
        # Ensure minimum 3 windows for short videos
        max_windows = max(3, total_frames // window_step)
        adjusted_step = max(total_frames // max_windows, window_step)
        
        windows = []
        start = 0
        while start < total_frames:
            end = min(start + window_size_frames, total_frames - 1)
            if end - start + 1 >= min_window_frames:
                windows.append({"start": start, "end": end, "id": len(windows)})
            start += adjusted_step
        
        self.logger.info(
            f"Created {len(windows)} adaptive sliding windows "
            f"(size={window_size_frames} frames, step={adjusted_step} frames)"
        )
        return windows

    async def process_video_sliding_window(
        self, 
        video_path: str, 
        vlm_analyze_function,
        window_size_sec: float = None,  # Changed to None to allow adaptive sizing
        window_overlap_sec: float = None,
        use_timestamps: bool = False,
        max_concurrent_vlm_calls: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Execute sliding window detection with binary search refinement.
        
        Phase 1: Process sliding windows to find action presence
        Phase 2: Binary search within positive windows for precise boundaries
        
        Returns frame results compatible with existing postprocessing.
        """
        # Get video metadata
        if is_macos_arm:
            if av is None:
                raise ImportError("PyAV is required on macOS ARM")
            container = av.open(video_path)
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            total_frames = stream.frames or 0
            container.close()
        else:
            if decord is None:
                raise ImportError("Decord is required on this platform")
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            del vr
        
        if total_frames == 0 or fps == 0:
            self.logger.error(f"Invalid video metadata: {total_frames} frames, {fps} fps")
            raise ValueError(f"Invalid video metadata: {total_frames} frames, {fps} fps")
        
        self.logger.info(f"Starting sliding window detection: {total_frames} frames @ {fps} fps")
        self.total_frames = total_frames
        self.api_calls_made = 0
        self.vlm_cache.clear()
        
        # Use adaptive window sizing if parameters are not explicitly provided
        if window_size_sec is not None and window_overlap_sec is not None:
            # Explicit parameters override adaptive sizing
            self.window_size_sec = window_size_sec
            self.window_overlap_sec = window_overlap_sec
            self.logger.info(f"Using explicit window parameters: {window_size_sec}s, overlap {window_overlap_sec}s")
        
        # Create semaphore to limit concurrent VLM calls
        vlm_semaphore = asyncio.Semaphore(max_concurrent_vlm_calls)
        
        # PHASE 1: Sliding window detection
        windows = self._create_sliding_windows(total_frames, fps)
        self.candidate_windows = await self._phase1_sliding_window_detection(
            video_path, vlm_analyze_function, vlm_semaphore, windows, fps, use_timestamps
        )
        
        # PHASE 2: Binary search within detected windows
        self.logger.info(f"Phase 2: Binary search for {len(self.candidate_windows)} candidate windows")
        processed_frame_data = await self._phase2_binary_search_within_windows(
            video_path, vlm_analyze_function, vlm_semaphore, fps, use_timestamps
        )
        
        frame_results = list(processed_frame_data.values())
        
        # Generate action segment results
        action_segments = self._generate_action_segments_from_windows(fps, use_timestamps)
        
        # Log statistics
        total_vlm_calls = self.api_calls_made
        total_possible_calls = total_frames * len(self.action_tags)
        efficiency = ((total_possible_calls - total_vlm_calls) / total_possible_calls * 100) if total_possible_calls > 0 else 0
        
        self.logger.info(
            f"Sliding window detection completed: {total_vlm_calls} API calls "
            f"(vs {total_possible_calls} full scan), {efficiency:.1f}% reduction"
        )
        
        if action_segments:
            self.logger.info(f"Detected {len(action_segments)} action segments:")
            for segment in action_segments:
                duration = segment['end_frame'] - segment['start_frame'] + 1
                self.logger.info(f"  {segment['action_tag']}: frames {segment['start_frame']}-{segment['end_frame']} ({duration} frames)")
        
        return frame_results

    async def _phase1_sliding_window_detection(
        self,
        video_path: str,
        vlm_analyze_function,
        vlm_semaphore: asyncio.Semaphore,
        windows: List[Dict[str, int]],
        fps: float,
        use_timestamps: bool
    ) -> List[Dict[str, Any]]:
        """Phase 1: Process each sliding window to detect action presence"""
        candidate_windows = []
        processed_frame_data = {}
        
        async def process_window(window: Dict[str, int]) -> List[Dict[str, Any]]:
            """Process a single window and return detected actions"""
            async with vlm_semaphore:
                try:
                    # Sample frames within this window
                    start, end = window["start"], window["end"]
                    sample_positions = []
                    
                    # Sample at start, middle, and end positions
                    positions = [start, (start + end) // 2, end]
                    
                    action_presence = {action: False for action in self.action_tags}
                    confidences = {action: 0.0 for action in self.action_tags}
                    
                    for frame_idx in positions:
                        # Check cache
                        vlm_cache_key = (video_path, frame_idx)
                        if vlm_cache_key in self.vlm_cache:
                            action_results = self.vlm_cache[vlm_cache_key]
                        else:
                            # Extract and analyze
                            frame_tensor = self.frame_extractor.extract_frame(video_path, frame_idx)
                            if frame_tensor is None:
                                continue
                            
                            frame_pil = self._convert_tensor_to_pil(frame_tensor)
                            if frame_pil is None:
                                continue
                            
                            action_results = await vlm_analyze_function(frame_pil)
                            self.api_calls_made += 1
                            self._cache_vlm_result(vlm_cache_key, action_results)
                        
                        # Track frame results
                        frame_identifier = float(frame_idx) / fps if use_timestamps else int(frame_idx)
                        processed_frame_data[frame_idx] = {
                            "frame_index": frame_identifier,
                            "frame_idx": frame_idx,
                            "action_results": action_results,
                            "actiondetection": [
                                (tag, conf) for tag, conf in action_results.items()
                                if conf >= self.threshold
                            ]
                        }
                        
                        # Update action presence based on any threshold pass
                        for action_tag in self.action_tags:
                            confidence = action_results.get(action_tag, 0.0)
                            if confidence >= self.threshold:
                                action_presence[action_tag] = True
                                confidences[action_tag] = max(confidences[action_tag], confidence)
                    
                    # Create window results
                    detected_actions = []
                    for action_tag in self.action_tags:
                        if action_presence[action_tag]:
                            detected_actions.append({
                                "action_tag": action_tag,
                                "window": window,
                                "confidence": confidences[action_tag],
                                "start_frame": window["start"],
                                "end_frame": window["end"]
                            })
                    
                    return detected_actions
                    
                except Exception as e:
                    self.logger.error(f"Window processing failed for window {window}: {e}")
                    return []
        
        # Process all windows concurrently
        window_tasks = [process_window(window) for window in windows]
        concurrent_results = await asyncio.gather(*window_tasks, return_exceptions=True)
        
        # Collect all detected actions
        for window_result in concurrent_results:
            if isinstance(window_result, list):
                candidate_windows.extend(window_result)
        
        self.logger.info(f"Phase 1 complete: {len(candidate_windows)} candidate windows detected")
        return candidate_windows

    async def _phase2_binary_search_within_windows(
        self,
        video_path: str,
        vlm_analyze_function,
        vlm_semaphore: asyncio.Semaphore,
        fps: float,
        use_timestamps: bool
    ) -> Dict[int, Dict[str, Any]]:
        """Phase 2: Binary search within each positive window for precise boundaries"""
        processed_frame_data = {}
        
        if not self.candidate_windows:
            self.logger.info("No candidate windows, skipping Phase 2")
            return processed_frame_data
        
        # Group windows by action tag for efficient processing
        action_windows = {}
        for window_info in self.candidate_windows:
            action_tag = window_info["action_tag"]
            if action_tag not in action_windows:
                action_windows[action_tag] = []
            action_windows[action_tag].append(window_info)
        
        # Create action ranges for each detected action
        self.action_ranges = []
        for action_tag, windows in action_windows.items():
            # Merge overlapping windows for this action
            windows.sort(key=lambda x: x["start_frame"])
            merged_window = {
                "start_frame": min(w["start_frame"] for w in windows),
                "end_frame": max(w["end_frame"] for w in windows)
            }
            
            action_range = ActionRange(
                start_frame=merged_window["start_frame"],
                end_frame=merged_window["end_frame"],
                action_tag=action_tag
            )
            action_range.confirmed_present = True
            action_range.start_found = merged_window["start_frame"]
            action_range.initiate_end_search(merged_window["end_frame"])
            action_range.reset_depth_for_end_search()
            self.action_ranges.append(action_range)
        
        return await self._perform_binary_search_within_ranges(
            video_path, vlm_analyze_function, vlm_semaphore, fps, use_timestamps, processed_frame_data
        )

    async def _perform_binary_search_within_ranges(
        self,
        video_path: str,
        vlm_analyze_function,
        vlm_semaphore: asyncio.Semaphore,
        fps: float,
        use_timestamps: bool,
        processed_frame_data: Dict[int, Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """Perform binary search refined within action ranges"""
        
        iteration = 0
        while self.has_unresolved_actions() and self._has_actions_within_depth_limit():
            iteration += 1
            
            # Guard against stalled searches
            for action_range in self.action_ranges:
                if not action_range.is_resolved() and action_range.searching_end:
                    if (action_range.end_search_start is not None and 
                        action_range.end_search_end is not None and
                        action_range.end_search_end - action_range.end_search_start <= 1):
                        action_range.is_stalled = True
            
            # Collect midpoints for binary search within ranges
            midpoints = self.midpoint_collector.collect_unique_midpoints(self.action_ranges)
            
            if not midpoints:
                break
            
            # Filter unprocessed midpoints
            unprocessed_midpoints = [idx for idx in midpoints if idx not in processed_frame_data]
            
            if not unprocessed_midpoints:
                # Re-apply existing results
                for frame_idx in midpoints:
                    if frame_idx in processed_frame_data:
                        action_results = processed_frame_data[frame_idx]["action_results"]
                        self.boundary_detector.update_action_boundaries(
                            self.action_ranges, frame_idx, action_results, self.total_frames
                        )
                continue
            
            # Process new midpoints
            async def process_midpoint_frame(frame_idx: int) -> Optional[Dict[str, Any]]:
                async with vlm_semaphore:
                    try:
                        vlm_cache_key = (video_path, frame_idx)
                        if vlm_cache_key in self.vlm_cache:
                            action_results = self.vlm_cache[vlm_cache_key]
                        else:
                            frame_tensor = self.frame_extractor.extract_frame(video_path, frame_idx)
                            if frame_tensor is None:
                                return None
                            
                            frame_pil = self._convert_tensor_to_pil(frame_tensor)
                            if frame_pil is None:
                                return None
                            
                            action_results = await vlm_analyze_function(frame_pil)
                            self.api_calls_made += 1
                            self._cache_vlm_result(vlm_cache_key, action_results)
                        
                        frame_identifier = float(frame_idx) / fps if use_timestamps else int(frame_idx)
                        return {
                            "frame_index": frame_identifier,
                            "frame_idx": frame_idx,
                            "action_results": action_results,
                            "actiondetection": [
                                (tag, confidence) for tag, confidence in action_results.items()
                                if confidence >= self.threshold
                            ]
                        }
                        
                    except Exception as e:
                        self.logger.error(f"Frame processing failed: {e}")
                        return None
            
            frame_tasks = [process_midpoint_frame(frame_idx) for frame_idx in unprocessed_midpoints]
            concurrent_results = await asyncio.gather(*frame_tasks, return_exceptions=True)
            
            for i, result in enumerate(concurrent_results):
                if isinstance(result, Exception) or result is None:
                    continue
                
                frame_idx = result["frame_idx"]
                action_results = result["action_results"]
                
                self.boundary_detector.update_action_boundaries(
                    self.action_ranges, frame_idx, action_results, self.total_frames
                )
                
                processed_frame_data[frame_idx] = result
        
        self.logger.info(f"Phase 2 complete: Processed {len(processed_frame_data)} frames in {iteration} iterations")
        return processed_frame_data

    def _generate_action_segments_from_windows(self, fps: float, use_timestamps: bool) -> List[Dict[str, Any]]:
        """Generate action segment results from refined window boundaries"""
        segments = []
        
        for action_range in self.action_ranges:
            if action_range.confirmed_present and action_range.start_found is not None:
                start_identifier = float(action_range.start_found) / fps if use_timestamps else int(action_range.start_found)
                
                end_frame = action_range.end_found if action_range.end_found is not None else action_range.start_found
                end_identifier = float(end_frame) / fps if use_timestamps else int(end_frame)
                
                segment = {
                    "action_tag": action_range.action_tag,
                    "start_frame": start_identifier,
                    "end_frame": end_identifier,
                    "duration": float(end_identifier - start_identifier),
                    "complete": action_range.is_resolved()
                }
                segments.append(segment)
        
        return segments

    # Backward compatibility - keep original methods working but mark as deprecated
    async def process_video_binary_search(self, *args, **kwargs):
        """Backward compatibility method - delegates to sliding window"""
        self.logger.warning("process_video_binary_search is deprecated, use process_video_sliding_window")
        return await self.process_video_sliding_window(*args, **kwargs)

    # Existing utility methods remain unchanged
    def _cache_vlm_result(self, cache_key: Tuple[str, int], action_results: Dict[str, float]) -> None:
        """Cache VLM analysis result with size limit management"""
        if len(self.vlm_cache) >= self.vlm_cache_size_limit:
            oldest_key = next(iter(self.vlm_cache))
            del self.vlm_cache[oldest_key]
            self.logger.debug(f"Evicted cached VLM result for frame {oldest_key[1]}")
        
        self.vlm_cache[cache_key] = action_results.copy()

    def clear_vlm_cache(self) -> None:
        """Clear the VLM analysis cache"""
        self.vlm_cache.clear()
        self.logger.debug("VLM analysis cache cleared")
    
    def _convert_tensor_to_pil(self, frame_tensor: torch.Tensor) -> Optional[Image.Image]:
        """Convert frame tensor to PIL Image for VLM processing"""
        try:
            if frame_tensor.is_cuda:
                frame_tensor = frame_tensor.cpu()
            
            if frame_tensor.dtype in (torch.float16, torch.float32):
                frame_np = frame_tensor.numpy()
                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).astype(np.uint8)
                else:
                    frame_np = frame_np.astype(np.uint8)
            else:
                frame_np = frame_tensor.numpy().astype(np.uint8)
            
            if frame_np.ndim == 3 and frame_np.shape[0] == 3:
                frame_np = np.transpose(frame_np, (1, 2, 0))
            
            return Image.fromarray(frame_np)
        except Exception as e:
            self.logger.error(f"Failed to convert tensor to PIL: {e}")
            return None

    def get_detected_segments(self) -> List[Dict[str, Any]]:
        """Get all detected action segments"""
        segments = []
        for action_range in self.action_ranges:
            if action_range.confirmed_present and action_range.start_found is not None and action_range.is_resolved():
                segments.append({
                    "action_tag": action_range.action_tag,
                    "start": action_range.start_found,
                    "end": action_range.end_found if action_range.end_found is not None else action_range.start_found
                })
        return segments
