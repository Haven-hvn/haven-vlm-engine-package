"""
Main engine implementing parallel binary search for action detection.
Replaces linear frame sampling with intelligent boundary detection.
"""

import asyncio
import logging
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
    Main engine implementing parallel binary search for action detection.
    Replaces linear frame sampling with intelligent boundary detection.
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
        self.total_frames = 0
        self.api_calls_made = 0
        
        # VLM analysis result caching
        self.vlm_cache: Dict[Tuple[str, int], Dict[str, float]] = {}
        self.vlm_cache_size_limit = 200  # Cache up to 200 VLM analysis results
        
        self.logger.info(f"ParallelBinarySearchEngine initialized for {len(self.action_tags)} actions")
    
    def initialize_search_ranges(self, total_frames: int) -> None:
        """Initialize search ranges for all actions"""
        self.total_frames = total_frames
        self.action_ranges = [
            ActionRange(
                start_frame=0,
                end_frame=total_frames - 1,
                action_tag=action_tag
            )
            for action_tag in self.action_tags
        ]
        self.api_calls_made = 0
        # Clear VLM cache for new video
        self.vlm_cache.clear()
        self.logger.info(f"Initialized search for {len(self.action_tags)} actions across {total_frames} frames")
    
    def has_unresolved_actions(self) -> bool:
        """Check if there are still actions being searched"""
        return any(not action_range.is_resolved() for action_range in self.action_ranges)
    
    async def process_video_binary_search(
        self, 
        video_path: str, 
        vlm_analyze_function,
        use_timestamps: bool = False,
        max_concurrent_vlm_calls: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Execute parallel binary search across the video with concurrent VLM processing.
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
        
        self.logger.info(f"Starting parallel binary search on video: {total_frames} frames @ {fps} fps")
        self.initialize_search_ranges(total_frames)
        
        frame_results = []
        processed_frames = set()
        
        # Create semaphore to limit concurrent VLM calls
        vlm_semaphore = asyncio.Semaphore(max_concurrent_vlm_calls)
        
        # Binary search loop
        while self.has_unresolved_actions():
            # Collect unique midpoints from all active searches
            midpoints = self.midpoint_collector.collect_unique_midpoints(self.action_ranges)
            
            if not midpoints:
                self.logger.warning("No midpoints collected but unresolved actions remain")
                break
            
            # Filter out already processed frames
            unprocessed_midpoints = [idx for idx in midpoints if idx not in processed_frames]
            
            if not unprocessed_midpoints:
                continue
            
            # Process all frames in this iteration concurrently
            async def process_single_frame(frame_idx: int) -> Optional[Dict[str, Any]]:
                """Process a single frame with VLM analysis and caching"""
                async with vlm_semaphore:
                    try:
                        # Check VLM cache first
                        vlm_cache_key = (video_path, frame_idx)
                        if vlm_cache_key in self.vlm_cache:
                            action_results = self.vlm_cache[vlm_cache_key]
                            self.logger.debug(f"VLM cache hit for frame {frame_idx}")
                        else:
                            # Extract frame
                            frame_tensor = self.frame_extractor.extract_frame(video_path, frame_idx)
                            if frame_tensor is None:
                                self.logger.warning(f"Failed to extract frame {frame_idx}")
                                return None
                            
                            # Convert to PIL for VLM processing
                            frame_pil = self._convert_tensor_to_pil(frame_tensor)
                            if frame_pil is None:
                                return None
                            
                            # Analyze frame with VLM
                            action_results = await vlm_analyze_function(frame_pil)
                            self.api_calls_made += 1
                            
                            # Cache the VLM analysis result
                            self._cache_vlm_result(vlm_cache_key, action_results)
                        
                        # Store frame result for postprocessing compatibility
                        frame_identifier = float(frame_idx) / fps if use_timestamps else int(frame_idx)
                        frame_result = {
                            "frame_index": frame_identifier,
                            "frame_idx": frame_idx,  # Keep original index for boundary updates
                            "action_results": action_results,
                            "actiondetection": [
                                (tag, confidence) for tag, confidence in action_results.items()
                                if confidence >= self.threshold
                            ]
                        }
                        
                        self.logger.debug(f"Processed frame {frame_idx}, API calls: {self.api_calls_made}")
                        return frame_result
                        
                    except Exception as e:
                        self.logger.error(f"VLM analysis failed for frame {frame_idx}: {e}")
                        return None
            
            # Execute all frame processing tasks concurrently
            self.logger.debug(f"Processing {len(unprocessed_midpoints)} frames concurrently")
            frame_tasks = [process_single_frame(frame_idx) for frame_idx in unprocessed_midpoints]
            concurrent_results = await asyncio.gather(*frame_tasks, return_exceptions=True)
            
            # Process results and update boundaries
            for i, result in enumerate(concurrent_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Frame processing failed: {result}")
                    continue
                
                if result is None:
                    continue
                
                result = cast(Dict[str, Any], result)
                frame_idx = result["frame_idx"]
                action_results = result["action_results"]
                
                # Update all action boundaries based on this frame's results
                self.boundary_detector.update_action_boundaries(
                    self.action_ranges, frame_idx, action_results, self.total_frames
                )
                
                # Store frame result (remove internal fields)
                frame_result = {
                    "frame_index": result["frame_index"],
                    "actiondetection": result["actiondetection"],
                    "frame_idx": result["frame_idx"],
                    "action_results": result["action_results"]
                }
                frame_results.append(frame_result)
                processed_frames.add(frame_idx)
        
        # Generate action segment results with start/end frame information
        action_segments = self._generate_action_segments(fps, use_timestamps)
        
        # Log performance metrics and action segment summary
        linear_calls = self.total_frames // max(1, int(fps * 0.5))  # Estimate linear approach
        efficiency = ((linear_calls - self.api_calls_made) / linear_calls * 100) if linear_calls > 0 else 0
        
        self.logger.info(
            f"Parallel binary search completed: {self.api_calls_made} API calls "
            f"(vs ~{linear_calls} linear), {efficiency:.1f}% reduction"
        )
        
        # Log detected action segments
        if action_segments:
            self.logger.info(f"Detected {len(action_segments)} action segments:")
            for segment in action_segments:
                duration = segment['end_frame'] - segment['start_frame'] + 1
                self.logger.info(f"  {segment['action_tag']}: frames {segment['start_frame']}-{segment['end_frame']} ({duration} frames)")
        
        return frame_results
    
    def _generate_action_segments(self, fps: float, use_timestamps: bool) -> List[Dict[str, Any]]:
        """Generate action segment results with start and end frame information"""
        segments = []
        
        for action_range in self.action_ranges:
            if action_range.confirmed_present and action_range.start_found is not None:
                start_identifier = float(action_range.start_found) / fps if use_timestamps else int(action_range.start_found)
                
                # Use end_found if available, otherwise use start_found (single frame action)
                end_frame = action_range.end_found if action_range.end_found is not None else action_range.start_found
                end_identifier = float(end_frame) / fps if use_timestamps else int(end_frame)
                
                segment = {
                    "action_tag": action_range.action_tag,
                    "start_frame": start_identifier,
                    "end_frame": end_identifier,
                    "duration": float(end_identifier - start_identifier),
                    "complete": action_range.end_found is not None
                }
                segments.append(segment)
        
        return segments
    
    def _cache_vlm_result(self, cache_key: Tuple[str, int], action_results: Dict[str, float]) -> None:
        """Cache VLM analysis result with size limit management"""
        if len(self.vlm_cache) >= self.vlm_cache_size_limit:
            # Remove oldest entry (simple FIFO eviction)
            oldest_key = next(iter(self.vlm_cache))
            del self.vlm_cache[oldest_key]
            self.logger.debug(f"Evicted cached VLM result for frame {oldest_key[1]} from {oldest_key[0]}")
        
        # Store a copy of the results to avoid reference issues
        self.vlm_cache[cache_key] = action_results.copy()
        self.logger.debug(f"Cached VLM result for frame {cache_key[1]} from {cache_key[0]}")
    
    def clear_vlm_cache(self) -> None:
        """Clear the VLM analysis cache"""
        self.vlm_cache.clear()
        self.logger.debug("VLM analysis cache cleared")
    
    def _convert_tensor_to_pil(self, frame_tensor: torch.Tensor) -> Optional[Image.Image]:
        """Convert frame tensor to PIL Image for VLM processing"""
        try:
            if frame_tensor.is_cuda:
                frame_tensor = frame_tensor.cpu()
            
            # Convert to numpy
            if frame_tensor.dtype in (torch.float16, torch.float32):
                frame_np = frame_tensor.numpy()
                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).astype(np.uint8)
                else:
                    frame_np = frame_np.astype(np.uint8)
            else:
                frame_np = frame_tensor.numpy().astype(np.uint8)
            
            # Ensure correct shape (H, W, C)
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
            if action_range.confirmed_present and action_range.start_found is not None and action_range.end_found is not None:
                segments.append({
                    "action_tag": action_range.action_tag,
                    "start": action_range.start_found,
                    "end": action_range.end_found
                })
        return segments
