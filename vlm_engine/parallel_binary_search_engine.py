"""
Main engine implementing hybrid linear scan + binary search for action detection.
Replaces pure binary search with a more reliable two-phase process.
"""

import asyncio
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, cast
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
import gc
from contextlib import contextmanager
import psutil

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
    Main engine implementing hybrid linear scan + binary search for action detection.
    Uses a two-phase approach: linear scan for action starts, then binary search for action ends.
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
        self.candidate_segments: List[Dict[str, Any]] = []  # Results from Phase 1
        self.total_frames = 0
        self.api_calls_made = 0
        
        # VLM analysis result caching
        self.vlm_cache: Dict[Tuple[str, int], Dict[str, float]] = {}
        self.vlm_cache_size_limit = 200  # Cache up to 200 VLM analysis results
        self.processed_frame_data_max = 500
        self.ram_log = True  # Set to False to disable RAM logging
        self.max_candidates = 100  # Warning threshold for lists
        
        # Optimization: Limit backward search in refinement
        self.max_backward_search_frames = 2000
        
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
    
    def _has_actions_within_depth_limit(self) -> bool:
        """Check if any actions are still within their depth limits"""
        return any(
            not action_range.is_resolved() and not action_range.has_reached_max_depth()
            for action_range in self.action_ranges
        )
    
    async def process_video_binary_search(
        self, 
        video_path: str, 
        vlm_analyze_function,
        frame_interval: float = 1.0, # Now represents the scan_frame_step in phase 1
        use_timestamps: bool = False,
        max_concurrent_vlm_calls: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid linear scan + binary search across the video with concurrent VLM processing.
        
        Phase 1: Linear scan to find candidate action starts  
        Phase 1.5: Binary-search backward to refine starts to exact first action frame
        Phase 2: Parallel binary search to refine action ends
        
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
        
        self.logger.info(f"Starting hybrid linear scan + binary search on video: {total_frames} frames @ {fps} fps")
        self.total_frames = total_frames
        self.api_calls_made = 0
        self.vlm_cache.clear()
        
        # Create semaphore to limit concurrent VLM calls
        vlm_semaphore = asyncio.Semaphore(max_concurrent_vlm_calls)
        
        # PHASE 1: Linear scan to find candidate action starts
        self.logger.info(f"Phase 1: Linear scan with frame step {frame_interval}")
        candidate_segments = await self._phase1_linear_scan(
            video_path, vlm_analyze_function, vlm_semaphore, total_frames, fps, use_timestamps, frame_interval
        )
        
        # PHASE 1.5 and 2: Overlap refinement and end search using queue
        if candidate_segments:
            self.logger.info(f"Phase 1.5+2: Overlapping refinement and end search for {len(candidate_segments)} candidates")
            
            # Queue for refined segments (producer: 1.5, consumer: 2)
            refinement_queue = asyncio.Queue()
            processed_frame_data = {}  # Collect results here
            
            # Producer task: Refine and enqueue segments
            async def producer():
                try:
                    refined_segments = []
                    for segment in candidate_segments:
                        try:
                            refined = await self._refine_single_start(segment, video_path, vlm_analyze_function, vlm_semaphore, total_frames)
                            refined_segments.append(refined)
                            # NEW: Force GC and log memory after each refinement
                            gc.collect()
                            if self.ram_log:
                                current_ram = psutil.Process().memory_info().rss / 1024**2
                                self.logger.info(f'RAM after refining segment {segment["action_tag"]}: {current_ram:.1f} MB')
                            self.frame_extractor.clear_cache()
                            await refinement_queue.put(refined)
                            self.logger.debug(f"Enqueued refined segment for {segment['action_tag']}")
                        except Exception as e:
                            self.logger.error(f"Error refining segment {segment['action_tag']}: {e}")
                            # Optionally continue or skip
                finally:
                    await refinement_queue.put(None)
            
            # Consumer task: Process enqueued segments for Phase 2
            async def consumer():
                while True:
                    segment = await refinement_queue.get()
                    if segment is None:
                        break
                    self.logger.debug(f"Consumer processing segment for {segment['action_tag']}")
                    # Process this segment's end search
                    segment_data = await self._phase2_process_single_segment(
                        video_path, vlm_analyze_function, vlm_semaphore, segment, total_frames, fps, use_timestamps
                    )
                    processed_frame_data.update(segment_data)
                    if len(processed_frame_data) > self.processed_frame_data_max:
                        oldest_key = next(iter(processed_frame_data))
                        del processed_frame_data[oldest_key]
                        self.logger.debug(f'Evicted old frame data for {oldest_key} to bound memory')
                        gc.collect()
                    gc.collect()
                return processed_frame_data
            
            # Run producer and consumer concurrently
            producer_task = asyncio.create_task(producer())
            consumer_task = asyncio.create_task(consumer())
            
            processed_frame_data = await consumer_task  # Back to simple await
            
            self.logger.info(f"Phase 1.5+2 complete: Processed {len(processed_frame_data)} frames")
        else:
            processed_frame_data = {}
            self.logger.info("No candidates, skipping Phase 1.5+2")

        frame_results = list(processed_frame_data.values())
        
        del processed_frame_data
        gc.collect()
        if self.ram_log:
            self.logger.info(f'Final RAM after clearing processed data: {psutil.Process().memory_info().rss / 1024**2:.1f} MB')
        
        # Generate action segment results with start/end frame information
        action_segments = self._generate_action_segments_from_candidates(fps, use_timestamps)
        
        # Log performance metrics and action segment summary
        linear_calls = self.total_frames // max(1, int(fps * 0.5))  # Estimate linear approach
        efficiency = ((linear_calls - self.api_calls_made) / linear_calls * 100) if linear_calls > 0 else 0
        
        self.logger.info(
            f"Hybrid scan completed: {self.api_calls_made} API calls "
            f"(vs ~{linear_calls} linear), {efficiency:.1f}% reduction"
        )
        
        # Log detected action segments
        if action_segments:
            self.logger.info(f"Detected {len(action_segments)} action segments:")
            for segment in action_segments:
                duration = segment['end_frame'] - segment['start_frame'] + 1
                self.logger.info(f"  {segment['action_tag']}: frames {segment['start_frame']}-{segment['end_frame']} ({duration} frames)")
        
        return frame_results
    
    async def _refine_starts_backward(
        self,
        video_path: str,
        vlm_analyze_function,
        vlm_semaphore: asyncio.Semaphore,
        candidate_segments: List[Dict[str, Any]],
        total_frames: int
    ) -> None:
        """
        Phase 1.5: Binary-search backward to refine starts to exact first action frame.
        
        For every segment, checks frames before the detected start to find the true
        first frame where the action is present.
        
        Args:
            video_path: Path to the video file
            vlm_analyze_function: VLM analysis function
            vlm_semaphore: Semaphore to limit concurrent VLM calls
            candidate_segments: List of detected segments from Phase 1 (modified in-place)
            total_frames: Total number of frames in the video
        """
        if not candidate_segments:
            return
            
        # Analyze frames concurrently and refine segments
        async def refine_segment_start(segment: Dict[str, Any]) -> None:
            """Refine start for a single candidate segment"""
            action_tag = segment["action_tag"]
            detected_start = segment["start_frame"]
            
            # Early exit for frame 0
            if detected_start <= 0:
                self.logger.debug(f"Segment {action_tag}: start is already at frame 0")
                return
            
            # Binary search range: [0, detected_start - 1]
            low = 0
            high = detected_start - 1
            refined_start = detected_start  # Start with detected
            
            self.logger.debug(f"Refining {action_tag}: search range [{low}, {high}]")
            
            while low <= high:
                mid = (low + high) // 2
                vlm_cache_key = (video_path, mid)
                
                async with vlm_semaphore:
                    try:
                        # Check cache first
                        if vlm_cache_key in self.vlm_cache:
                            action_results = self.vlm_cache[vlm_cache_key]
                            self.logger.debug(f"VLM cache hit for refinement frame {mid}")
                        else:
                            # Extract and analyze frame
                            with self.temp_frame(video_path, mid, phase='Phase 1.5') as (frame_tensor, frame_pil):
                                if frame_pil is None:
                                    low = mid + 1
                                    continue
                                action_results = await vlm_analyze_function(frame_pil)
                                self.api_calls_made += 1
                                self._cache_vlm_result(vlm_cache_key, action_results)
                        
                        # Check if action is present
                        confidence = action_results.get(action_tag, 0.0)
                        is_present = confidence >= self.threshold
                        
                        if is_present:
                            # Action present - move left to find earlier frame
                            refined_start = mid
                            high = mid - 1
                            self.logger.debug(f"Action present at {mid}, new refined_start={refined_start}")
                        else:
                            # Action absent - move right
                            low = mid + 1
                    
                    except Exception as e:
                        self.logger.error(f"Error refining {action_tag} at frame {mid}: {e}")
                        low = mid + 1  # Skip problematic frame
            
            # Update segment with refined start
            original_start = segment["start_frame"]
            segment["start_frame"] = refined_start
            self.logger.debug(f"{action_tag}: refined start from {original_start} to {refined_start}")
        
        # Refine all segments concurrently
        refinement_tasks = [refine_segment_start(segment) for segment in candidate_segments]
        await asyncio.gather(*refinement_tasks)
    
    async def _phase1_linear_scan(
        self,
        video_path: str,
        vlm_analyze_function,
        vlm_semaphore: asyncio.Semaphore,
        total_frames: int,
        fps: float,
        use_timestamps: bool,
        frame_interval: float # Frame interval for linear scan
    ) -> List[Dict[str, Any]]:
        """
        Phase 1: Linear scan to find candidate action starts.
        
        Process frames at regular intervals to detect when actions transition from absent to present.
        """
        candidate_segments = []
        processed_frame_data = {}
        
        # Track last known state for each action to detect transitions
        last_action_states = {action_tag: False for action_tag in self.action_tags}
        
        # Sample frames at regular intervals. Convert frame_interval from seconds to frames.
        # Ensure frame_step is at least 1 frame.
        frame_step_frames = max(1, int(frame_interval * fps)) * 30
        scan_frames = list(range(0, total_frames, frame_step_frames))
        if scan_frames[-1] != total_frames - 1:
            scan_frames.append(total_frames - 1)  # Always include the last frame
        
        self.logger.info(f"Phase 1: Scanning {len(scan_frames)} frames (every {frame_step_frames} frames)")
        
        # Process frames concurrently
        async def process_scan_frame(frame_idx: int) -> Optional[Dict[str, Any]]:
            """Process a single frame in the linear scan"""
            async with vlm_semaphore:
                try:
                    # Check VLM cache first
                    vlm_cache_key = (video_path, frame_idx)
                    if vlm_cache_key in self.vlm_cache:
                        action_results = self.vlm_cache[vlm_cache_key]
                        self.logger.debug(f"VLM cache hit for frame {frame_idx}")
                    else:
                        # Extract frame
                        with self.temp_frame(video_path, frame_idx, phase='Phase 1') as (frame_tensor, frame_pil):
                            if frame_pil is None:
                                return None
                            action_results = await vlm_analyze_function(frame_pil)
                            self.api_calls_made += 1
                            self._cache_vlm_result(vlm_cache_key, action_results)
                    
                    # Store frame result for postprocessing compatibility
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
                    self.logger.error(f"VLM analysis failed for frame {frame_idx}: {e}")
                    return None
        
        # Process all scan frames concurrently
        frame_tasks = [process_scan_frame(frame_idx) for frame_idx in scan_frames]
        concurrent_results = await asyncio.gather(*frame_tasks, return_exceptions=True)
        
        # Analyze results to detect action transitions
        for i, result in enumerate(concurrent_results):
            if isinstance(result, Exception) or result is None:
                continue
            assert isinstance(result, dict)
            frame_idx = result["frame_idx"]
            action_results = result["action_results"]
            
            # Store processed frame data
            processed_frame_data[frame_idx] = result
            
            # NEW: Evict oldest if exceeding limit
            if len(processed_frame_data) > self.processed_frame_data_max:
                oldest_key = next(iter(processed_frame_data))
                del processed_frame_data[oldest_key]
                self.logger.debug(f'Evicted old frame data for {oldest_key} to bound memory')
                gc.collect()

            # NEW: Periodic GC every 50 frames to release memory aggressively
            if i % 50 == 0:
                gc.collect()
                self.logger.debug(f'Periodic GC after processing frame {frame_idx}')

            # Check for action transitions (absent -> present)
            for action_tag in self.action_tags:
                confidence = action_results.get(action_tag, 0.0)
                is_present = confidence >= self.threshold
                
                # If action transitioned from absent to present, mark as candidate start
                if not last_action_states[action_tag] and is_present:
                    candidate_segments.append({
                        "action_tag": action_tag,
                        "start_frame": frame_idx,
                        "end_frame": None,  # To be determined in Phase 2
                        "confidence": confidence
                    })
                    self.logger.debug(f"Found candidate start for '{action_tag}' at frame {frame_idx}")
                
                # Update last known state
                last_action_states[action_tag] = is_present
        
        # Store candidate segments for Phase 2
        self.candidate_segments = candidate_segments
        
        self.logger.info(f"Phase 1 complete: Found {len(candidate_segments)} candidate action segments")
        
        # NEW: Clear frame extractor cache after Phase 1
        self.frame_extractor.clear_cache()
        gc.collect()
        self.logger.debug('Cleared frame cache and forced GC after Phase 1')
        if len(candidate_segments) > self.max_candidates:
            self.logger.warning(f'Too many candidate segments ({len(candidate_segments)}), consider adjusting sampling or filtering')
        return candidate_segments
    
    async def _phase2_binary_search(
        self,
        video_path: str,
        vlm_analyze_function,
        vlm_semaphore: asyncio.Semaphore,
        candidate_segments: List[Dict[str, Any]],
        total_frames: int,
        fps: float,
        use_timestamps: bool
    ) -> Dict[int, Dict[str, Any]]:
        """
        Phase 2: Parallel binary search to refine action ends.
        
        For each candidate segment, perform binary search to find the exact end frame.
        """
        processed_frame_data = {}
        
        if not candidate_segments:
            self.logger.info("No candidate segments found, skipping Phase 2")
            return processed_frame_data
        
        # Initialize action ranges for binary search
        self.action_ranges = []
        for segment in candidate_segments:
            action_range = ActionRange(
                start_frame=segment["start_frame"],
                end_frame=total_frames - 1,
                action_tag=segment["action_tag"]
            )
            # Mark as confirmed present and set start_found
            action_range.confirmed_present = True
            action_range.start_found = segment["start_frame"]
            action_range.initiate_end_search(total_frames)
            # Reset depth counter for end search
            action_range.reset_depth_for_end_search()
            self.action_ranges.append(action_range)
        
        # Log per-action depth information
        for action_range in self.action_ranges:
            search_range = action_range.end_search_end - action_range.end_search_start + 1 if action_range.end_search_end and action_range.end_search_start else 0
            self.logger.info(f"Action '{action_range.action_tag}': max_depth={action_range.max_depth}, search_range={search_range}")
        
        iteration = 0
        
        while self.has_unresolved_actions() and self._has_actions_within_depth_limit():
            iteration += 1
            
            # Guard against stalled searches
            for action_range in self.action_ranges:
                if not action_range.is_resolved() and action_range.searching_end:
                    if (action_range.end_search_start is not None and 
                        action_range.end_search_end is not None and
                        action_range.end_search_end - action_range.end_search_start <= 1):
                        self.logger.debug(f"Binary search window collapsed for {action_range.action_tag}, resolving")
                        action_range.is_stalled = True
            
            # Collect midpoints for binary search
            midpoints = self.midpoint_collector.collect_unique_midpoints(self.action_ranges)
            
            if not midpoints:
                self.logger.debug("No midpoints to process, ending Phase 2")
                break
            
            # Filter out already processed midpoints
            unprocessed_midpoints = [idx for idx in midpoints if idx not in processed_frame_data]
            
            if not unprocessed_midpoints:
                # Re-apply existing results to advance search ranges
                for frame_idx in midpoints:
                    if frame_idx in processed_frame_data:
                        action_results = processed_frame_data[frame_idx]["action_results"]
                        self.boundary_detector.update_action_boundaries(
                            self.action_ranges, frame_idx, action_results, total_frames
                        )
                continue
            
            # Process unprocessed midpoints
            async def process_midpoint_frame(frame_idx: int) -> Optional[Dict[str, Any]]:
                """Process a single frame in the binary search"""
                async with vlm_semaphore:
                    try:
                        # Check VLM cache first
                        vlm_cache_key = (video_path, frame_idx)
                        if vlm_cache_key in self.vlm_cache:
                            action_results = self.vlm_cache[vlm_cache_key]
                            self.logger.debug(f"VLM cache hit for frame {frame_idx}")
                        else:
                            # Extract frame
                            with self.temp_frame(video_path, frame_idx, phase='Phase 2') as (frame_tensor, frame_pil):
                                if frame_pil is None:
                                    return None
                            
                            # Analyze frame with VLM
                            action_results = await vlm_analyze_function(frame_pil)
                            self.api_calls_made += 1
                            
                            # Cache the VLM analysis result
                            self._cache_vlm_result(vlm_cache_key, action_results)
                        
                        # Store frame result for postprocessing compatibility
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
                        self.logger.error(f"VLM analysis failed for frame {frame_idx}: {e}")
                        return None
            
            # Process all midpoint frames concurrently
            frame_tasks = [process_midpoint_frame(frame_idx) for frame_idx in unprocessed_midpoints]
            concurrent_results = await asyncio.gather(*frame_tasks, return_exceptions=True)
            
            # Process results and update boundaries
            for i, result in enumerate(concurrent_results):
                if isinstance(result, Exception) or result is None:
                    continue
                assert isinstance(result, dict)
                frame_idx = result["frame_idx"]
                action_results = result["action_results"]
                
                # Update action boundaries based on this frame's results
                self.boundary_detector.update_action_boundaries(
                    self.action_ranges, frame_idx, action_results, total_frames
                )
                
                # Store frame result
                processed_frame_data[frame_idx] = result
        
        # Log per-action completion status
        for action_range in self.action_ranges:
            if action_range.is_resolved():
                completion_reason = "boundary found"
            elif action_range.has_reached_max_depth():
                completion_reason = f"max depth {action_range.max_depth} reached"
            else:
                completion_reason = "still searching"
            
            self.logger.info(f"Action '{action_range.action_tag}': depth {action_range.current_depth}/{action_range.max_depth}, {completion_reason}")
        
        self.logger.info(f"Phase 2 complete: Processed {len(processed_frame_data)} frames in {iteration} iterations")
        if len(self.action_ranges) > self.max_candidates:
            self.logger.warning(f'Too many action ranges ({len(self.action_ranges)}), potential performance issue')
        return processed_frame_data
    
    def _generate_action_segments_from_candidates(self, fps: float, use_timestamps: bool) -> List[Dict[str, Any]]:
        """Generate action segment results from candidate segments and refined boundaries"""
        segments = []
        
        for action_range in self.action_ranges:
            if action_range.confirmed_present and action_range.start_found is not None:
                start_identifier = float(action_range.start_found) / fps if use_timestamps else int(action_range.start_found)
                
                # Use end_found if available, otherwise use start_found for single-frame actions
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
    
    def _generate_action_segments(self, fps: float, use_timestamps: bool) -> List[Dict[str, Any]]:
        """Generate action segment results with start and end frame information"""
        segments = []
        
        for action_range in self.action_ranges:
            if action_range.confirmed_present and action_range.start_found is not None:
                start_identifier = float(action_range.start_found) / fps if use_timestamps else int(action_range.start_found)
                
                # If end_found is not set, but the action is resolved, it's a single-frame action.
                end_frame = action_range.end_found if action_range.end_found is not None else action_range.start_found
                end_identifier = float(end_frame) / fps if use_timestamps else int(end_frame)
                
                segment = {
                    "action_tag": action_range.action_tag,
                    "start_frame": start_identifier,
                    "end_frame": end_identifier,
                    "duration": float(end_identifier - start_identifier),
                    "complete": action_range.is_resolved() # A segment is complete if the range is resolved.
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
        gc.collect()
    
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
            if action_range.confirmed_present and action_range.start_found is not None and action_range.is_resolved():
                segments.append({
                    "action_tag": action_range.action_tag,
                    "start": action_range.start_found,
                    "end": action_range.end_found if action_range.end_found is not None else action_range.start_found
                })
        return segments

    async def _refine_single_start(
        self,
        segment: Dict[str, Any],
        video_path: str,
        vlm_analyze_function,
        vlm_semaphore: asyncio.Semaphore,
        total_frames: int
    ) -> Dict[str, Any]:
        """Refine the start frame of a candidate segment using backward binary search."""
        action_tag = segment["action_tag"]
        detected_start = segment["start_frame"]
        
        # Limit the backward search range to avoid searching the entire video
        refine_start = max(0, detected_start - self.max_backward_search_frames)
        refine_range = ActionRange(
            refine_start,
            detected_start - 1,
            action_tag
        )
        self.logger.debug(f"Refining {action_tag}: search range [{refine_start}, {detected_start - 1}], max_depth={refine_range.max_depth}")
        
        refined_start = refine_start  # Initialize to start of range
        
        while refine_range.start_frame <= refine_range.end_frame:
            mid = refine_range.get_midpoint()
            if mid is None:
                break
            
            async with vlm_semaphore:
                try:
                    # Check cache first
                    vlm_cache_key = (video_path, mid)
                    if vlm_cache_key in self.vlm_cache:
                        action_results = self.vlm_cache[vlm_cache_key]
                        self.logger.debug(f"VLM cache hit for refinement frame {mid}")
                    else:
                        with self.temp_frame(video_path, mid, phase='Phase 1.5') as (frame_tensor, frame_pil):
                            if frame_pil is None:
                                refine_range.start_frame = mid + 1
                                continue
                            action_results = await vlm_analyze_function(frame_pil)
                            self.api_calls_made += 1
                            self._cache_vlm_result(vlm_cache_key, action_results)
                    
                    confidence = action_results.get(action_tag, 0.0)
                    is_present = confidence >= self.threshold
                    
                    if is_present:
                        # Action is present, true start is at or before this frame
                        refined_start = mid  # Update potential start
                        refine_range.end_frame = mid - 1  # Search left half
                        self.logger.debug(f"{action_tag} present at {mid}, searching left")
                    else:
                        # Action absent, true start is after this frame
                        refine_range.start_frame = mid + 1  # Search right half
                        self.logger.debug(f"{action_tag} absent at {mid}, searching right")
                    
                    # NEW: Periodic GC every 5 depth levels to release memory
                    refine_range.current_depth += 1
                    if refine_range.current_depth % 5 == 0:
                        gc.collect()
                        self.logger.debug(f'Periodic GC at depth {refine_range.current_depth} for {action_tag} refinement')
                
                except Exception as e:
                    self.logger.error(f"Error refining {action_tag} at frame {mid}: {e}")
                    refine_range.start_frame = mid + 1
        
        segment["start_frame"] = refined_start
        self.logger.debug(f"{action_tag}: refined start from {detected_start} to {refined_start}")
        return segment

        # NEW: Clear frame cache after refinement
        self.frame_extractor.clear_cache()
        gc.collect()
        self.logger.debug(f'Cleared frame cache after refining {action_tag}')

    async def _phase2_process_single_segment(
        self,
        video_path: str,
        vlm_analyze_function,
        vlm_semaphore: asyncio.Semaphore,
        segment: Dict[str, Any],
        total_frames: int,
        fps: float,
        use_timestamps: bool
    ) -> Dict[int, Dict[str, Any]]:
        """Process end search for a single segment (subset of _phase2_binary_search)."""
        # (Adapt logic from _phase2_binary_search for one segment: Initialize single ActionRange, run the while loop, return processed_frame_data for this segment)
        processed_frame_data = {}
        
        action_range = ActionRange(
            start_frame=segment["start_frame"],
            end_frame=total_frames - 1,
            action_tag=segment["action_tag"]
        )
        action_range.confirmed_present = True
        action_range.start_found = segment["start_frame"]
        action_range.initiate_end_search(total_frames)
        action_range.reset_depth_for_end_search()
        
        self.logger.debug(f"Starting Phase 2 for {action_range.action_tag} with end range [{action_range.end_search_start}, {action_range.end_search_end}]")
        
        iteration = 0
        while not action_range.is_resolved() and not action_range.has_reached_max_depth():
            iteration += 1
            
            if action_range.end_search_start is not None and action_range.end_search_end is not None and action_range.end_search_end - action_range.end_search_start <= 1:
                self.logger.debug(f"Window collapsed for {action_range.action_tag}, resolving")
                action_range.is_stalled = True
                break
            
            midpoint = action_range.get_midpoint()
            if midpoint is None:
                break
            self.logger.debug(f"Iteration {iteration} for {action_range.action_tag}, midpoint={midpoint}")
            
            if midpoint in processed_frame_data:
                action_results = processed_frame_data[midpoint]["action_results"]
            else:
                async with vlm_semaphore:
                    with self.temp_frame(video_path, midpoint, phase='Phase 2') as (frame_tensor, frame_pil):
                        if frame_pil is None:
                            self.logger.warning(f"Failed to extract frame {midpoint} for {action_range.action_tag}, skipping")
                            continue
            
            action_results = await vlm_analyze_function(frame_pil)
            self.api_calls_made += 1
            self._cache_vlm_result((video_path, midpoint), action_results)

            # Update boundaries with the results
            self.boundary_detector.update_action_boundaries([action_range], midpoint, action_results, total_frames)

            # Store frame result
            frame_identifier = float(midpoint) / fps if use_timestamps else int(midpoint)
            processed_frame_data[midpoint] = {
                "frame_index": frame_identifier,
                "frame_idx": midpoint,
                "action_results": action_results,
                "actiondetection": [
                    (tag, confidence) for tag, confidence in action_results.items()
                    if confidence >= self.threshold
                ]
            }
        
        self.logger.debug(f"Phase 2 ended for {action_range.action_tag}: resolved={action_range.is_resolved()}, stalled={action_range.is_stalled}, depth={action_range.current_depth}, end_found={action_range.end_found}")
        
        # NEW: Handle cases where no midpoints were processed (small/collapsed range)
        if action_range.end_found is None and action_range.searching_end:
            if action_range.end_search_start is not None and action_range.end_search_end is not None:
                if action_range.end_search_start >= action_range.end_search_end:
                    # Collapsed or single-frame action
                    action_range.end_found = action_range.end_search_end
                    self.logger.debug(f"Set end_found to {action_range.end_found} for small range in {action_range.action_tag}")
        
        return processed_frame_data

    @contextmanager
    def temp_frame(self, video_path, frame_idx, phase: str = ''):
        frame_tensor = self.frame_extractor.extract_frame(video_path, frame_idx)
        if frame_tensor is None:
            yield None, None
            return
        frame_pil = self._convert_tensor_to_pil(frame_tensor)
        yield frame_tensor, frame_pil
        del frame_tensor, frame_pil
        gc.collect()
        log_msg = f'RAM after release for frame {frame_idx}'
        if phase:
            log_msg += f' in {phase}'
        log_msg += f': {psutil.Process().memory_info().rss / 1024**2:.1f} MB'
        self.logger.info(log_msg)