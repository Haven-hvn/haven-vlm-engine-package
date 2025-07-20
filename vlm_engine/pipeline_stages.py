"""
Pipeline Stages for Binary Search Video Processing

This module contains the modular pipeline stages that replace the monolithic
ParallelBinarySearchEngine with a clean, event-driven architecture.

Stages:
1. MetadataExtractionStage - Video analysis and initialization
2. CandidateProposalStage - Linear scan for action candidates  
3. StartRefinementStage - Binary search for precise start boundaries
4. EndDeterminationStage - Binary search for precise end boundaries
5. ResultCompilationStage - Aggregate results into final format
"""

import asyncio
import logging
import gc
import psutil
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Callable, Tuple
from PIL import Image
import torch
import numpy as np

from .models import Model
from .config_models import ModelConfig
from .async_utils import QueueItem, ItemFuture
from .action_range import ActionRange
from .adaptive_midpoint_collector import AdaptiveMidpointCollector
from .action_boundary_detector import ActionBoundaryDetector
from .video_frame_extractor import VideoFrameExtractor
from .preprocessing import is_macos_arm, crop_black_bars_lr

# Import video processing backends
if is_macos_arm:
    import av
else:
    import decord
    decord.bridge.set_bridge('torch')


class BasePipelineStage(Model):
    """Base class for all binary search pipeline stages"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.logger = logging.getLogger("logger")
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half_precision = getattr(config, 'use_half_precision', True)
        
    async def worker_function(self, data: List[QueueItem]) -> None:
        """Process queue items for this stage"""
        for item in data:
            try:
                await self.process_item(item)
            except Exception as e:
                self.logger.error(f"Error in {self.__class__.__name__}: {e}", exc_info=True)
                item.item_future.set_exception(e)
    
    async def process_item(self, item: QueueItem) -> None:
        """Override this method in each stage"""
        raise NotImplementedError("Each stage must implement process_item")


class MetadataExtractionStage(BasePipelineStage):
    """
    Stage 1: Extract video metadata and initialize search parameters
    
    Inputs: video_path, action_tags, threshold, frame_interval, use_timestamps, max_concurrent_vlm_calls
    Outputs: video_metadata (contains fps, total_frames, action_ranges, vlm_semaphore)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
    async def process_item(self, item: QueueItem) -> None:
        """Extract video metadata and initialize search state"""
        item_future = item.item_future
        
        # Extract inputs
        video_path = item_future[item.input_names[0]]
        action_tags = item_future[item.input_names[1]]
        threshold = item_future[item.input_names[2]]
        frame_interval = item_future[item.input_names[3]]
        use_timestamps = item_future[item.input_names[4]]
        max_concurrent_vlm_calls = item_future[item.input_names[5]]
        
        self.logger.info(f"Extracting metadata for video: {video_path}")
        
        try:
            # Extract video metadata using the same logic as ParallelBinarySearchEngine
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
                raise ValueError(f"Invalid video metadata: {total_frames} frames, {fps} fps")
            
            # Initialize action ranges for binary search
            action_ranges = [
                ActionRange(
                    start_frame=0,
                    end_frame=total_frames - 1,
                    action_tag=action_tag
                )
                for action_tag in action_tags
            ]
            
            # Create VLM semaphore for concurrency control
            vlm_semaphore = asyncio.Semaphore(max_concurrent_vlm_calls)
            
            # Initialize shared components
            frame_extractor = VideoFrameExtractor(self.device, self.use_half_precision)
            vlm_cache = {}  # Shared VLM result cache
            
            # Package metadata for next stages
            video_metadata = {
                "video_path": video_path,
                "fps": fps,
                "total_frames": total_frames,
                "action_tags": action_tags,
                "action_ranges": action_ranges,
                "threshold": threshold,
                "frame_interval": frame_interval,
                "use_timestamps": use_timestamps,
                "vlm_semaphore": vlm_semaphore,
                "frame_extractor": frame_extractor,
                "vlm_cache": vlm_cache,
                "api_calls_made": 0,
                "processed_frame_data": {}
            }
            
            self.logger.info(f"Video metadata extracted: {total_frames} frames @ {fps} fps, {len(action_tags)} actions")
            
            # Output metadata to next stage
            await item_future.set_data(item.output_names[0], video_metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to extract video metadata: {e}")
            raise


class CandidateProposalStage(BasePipelineStage):
    """
    Stage 2: Linear scan to find candidate action starts (Phase 1)
    
    Inputs: video_metadata, vlm_analyze_function
    Outputs: candidate_segments
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
    async def process_item(self, item: QueueItem) -> None:
        """Perform linear scan to find candidate action starts"""
        item_future = item.item_future
        
        # Extract inputs
        video_metadata = item_future[item.input_names[0]]
        vlm_analyze_function = item_future[item.input_names[1]]
        
        # Unpack metadata
        video_path = video_metadata["video_path"]
        fps = video_metadata["fps"]
        total_frames = video_metadata["total_frames"]
        action_tags = video_metadata["action_tags"]
        frame_interval = video_metadata["frame_interval"]
        use_timestamps = video_metadata["use_timestamps"]
        vlm_semaphore = video_metadata["vlm_semaphore"]
        frame_extractor = video_metadata["frame_extractor"]
        vlm_cache = video_metadata["vlm_cache"]
        
        self.logger.info(f"Phase 1: Linear scan with frame step {frame_interval}")
        
        try:
            candidate_segments = await self._phase1_linear_scan(
                video_path, vlm_analyze_function, vlm_semaphore, total_frames, fps, 
                use_timestamps, frame_interval, action_tags, frame_extractor, vlm_cache, video_metadata
            )
            
            self.logger.info(f"Phase 1 complete: Found {len(candidate_segments)} candidate segments")
            
            # Update metadata with results
            video_metadata["candidate_segments"] = candidate_segments
            
            # Output to next stage
            await item_future.set_data(item.output_names[0], video_metadata)
            
        except Exception as e:
            self.logger.error(f"Failed in candidate proposal stage: {e}")
            raise
    
    async def _phase1_linear_scan(
        self,
        video_path: str,
        vlm_analyze_function,
        vlm_semaphore: asyncio.Semaphore,
        total_frames: int,
        fps: float,
        use_timestamps: bool,
        frame_interval: float,
        action_tags: List[str],
        frame_extractor: VideoFrameExtractor,
        vlm_cache: Dict,
        video_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """
        Phase 1: Linear scan to find candidate action starts.
        
        Process frames at regular intervals to detect when actions transition from absent to present.
        """
        candidate_segments = []
        processed_frame_data = video_metadata["processed_frame_data"]
        
        # Track last known state for each action to detect transitions
        last_action_states = {action_tag: False for action_tag in action_tags}
        
        # Calculate frame step for linear scan
        frame_step = max(1, int(fps * frame_interval))
        scan_frames = list(range(0, total_frames, frame_step))
        
        self.logger.info(f"Linear scan: processing {len(scan_frames)} frames with step {frame_step}")
        
        for frame_idx in scan_frames:
            async with vlm_semaphore:
                try:
                    # Check VLM cache first
                    vlm_cache_key = (video_path, frame_idx)
                    if vlm_cache_key in vlm_cache:
                        action_results = vlm_cache[vlm_cache_key]
                        self.logger.debug(f"VLM cache hit for frame {frame_idx}")
                    else:
                        # Extract frame
                        with self._temp_frame(frame_extractor, video_path, frame_idx, 'Phase 1') as (frame_tensor, frame_pil):
                            if frame_pil is None:
                                continue
                        
                        # Analyze frame with VLM
                        action_results = await vlm_analyze_function(frame_pil)
                        video_metadata["api_calls_made"] += 1
                        
                        # Cache the VLM analysis result
                        self._cache_vlm_result(vlm_cache, vlm_cache_key, action_results)
                    
                    # Store frame result
                    frame_result = {
                        "frame_idx": frame_idx,
                        "action_results": action_results,
                        "timestamp": frame_idx / fps if use_timestamps else frame_idx
                    }
                    processed_frame_data[frame_idx] = frame_result
                    
                    # Check for action transitions (absent -> present)
                    for action_tag in action_tags:
                        confidence = action_results.get(action_tag, 0.0)
                        is_present = confidence >= video_metadata["threshold"]
                        was_present = last_action_states[action_tag]
                        
                        # Detect transition from absent to present
                        if is_present and not was_present:
                            candidate_segment = {
                                "action_tag": action_tag,
                                "start_frame": frame_idx,
                                "confidence": confidence,
                                "timestamp": frame_idx / fps if use_timestamps else frame_idx
                            }
                            candidate_segments.append(candidate_segment)
                            self.logger.debug(f"Candidate start detected: {action_tag} at frame {frame_idx} (confidence: {confidence:.3f})")
                        
                        # Update state
                        last_action_states[action_tag] = is_present
                
                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_idx} in Phase 1: {e}")
                    continue
        
        return candidate_segments
    
    @contextmanager
    def _temp_frame(self, frame_extractor, video_path, frame_idx, phase: str = ''):
        """Temporary frame extraction with cleanup"""
        frame_tensor = frame_extractor.extract_frame(video_path, frame_idx)
        if frame_tensor is None:
            yield None, None
            return
        frame_pil = self._convert_tensor_to_pil(frame_tensor)
        yield frame_tensor, frame_pil
        del frame_tensor, frame_pil
        gc.collect()
    
    def _convert_tensor_to_pil(self, frame_tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image"""
        if frame_tensor.dtype == torch.float16:
            frame_tensor = frame_tensor.float()
        
        # Ensure tensor is in [0, 1] range
        if frame_tensor.max() <= 1.0:
            frame_tensor = frame_tensor * 255.0
        
        frame_tensor = frame_tensor.clamp(0, 255).byte()
        
        # Convert to numpy and PIL
        frame_np = frame_tensor.cpu().numpy()
        if frame_np.ndim == 3 and frame_np.shape[0] == 3:
            frame_np = frame_np.transpose(1, 2, 0)
        
        return Image.fromarray(frame_np)
    
    def _cache_vlm_result(self, vlm_cache: Dict, cache_key: Tuple, action_results: Dict[str, float]):
        """Cache VLM analysis result with size limit"""
        vlm_cache[cache_key] = action_results

        # Implement LRU-style cache eviction if needed
        if len(vlm_cache) > 200:  # Cache size limit
            # Remove oldest entries
            oldest_keys = list(vlm_cache.keys())[:50]
            for key in oldest_keys:
                del vlm_cache[key]


class StartRefinementStage(BasePipelineStage):
    """
    Stage 3: Binary search backward to refine starts to exact first action frame (Phase 1.5)

    Inputs: video_metadata (with candidate_segments), vlm_analyze_function
    Outputs: video_metadata (with refined candidate_segments)
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.max_backward_search_frames = 2000  # Limit backward search

    async def process_item(self, item: QueueItem) -> None:
        """Refine candidate segment start boundaries using binary search"""
        item_future = item.item_future

        # Extract inputs
        video_metadata = item_future[item.input_names[0]]
        vlm_analyze_function = item_future[item.input_names[1]]

        candidate_segments = video_metadata.get("candidate_segments", [])

        if not candidate_segments:
            self.logger.info("No candidate segments to refine, skipping Phase 1.5")
            await item_future.set_data(item.output_names[0], video_metadata)
            return

        self.logger.info(f"Phase 1.5: Refining start boundaries for {len(candidate_segments)} candidates")

        try:
            await self._refine_starts_backward(
                video_metadata, vlm_analyze_function, candidate_segments
            )

            self.logger.info(f"Phase 1.5 complete: Refined {len(candidate_segments)} segment starts")

            # Output refined metadata
            await item_future.set_data(item.output_names[0], video_metadata)

        except Exception as e:
            self.logger.error(f"Failed in start refinement stage: {e}")
            raise

    async def _refine_starts_backward(
        self,
        video_metadata: Dict,
        vlm_analyze_function,
        candidate_segments: List[Dict[str, Any]]
    ) -> None:
        """
        Phase 1.5: Binary-search backward to refine starts to exact first action frame.

        For every segment, checks frames before the detected start to find the true
        first frame where the action is present.
        """
        video_path = video_metadata["video_path"]
        total_frames = video_metadata["total_frames"]
        threshold = video_metadata["threshold"]
        vlm_semaphore = video_metadata["vlm_semaphore"]
        frame_extractor = video_metadata["frame_extractor"]
        vlm_cache = video_metadata["vlm_cache"]

        async def refine_segment_start(segment: Dict[str, Any]) -> None:
            """Refine the start boundary of a single segment"""
            action_tag = segment["action_tag"]
            detected_start = segment["start_frame"]

            # Define search range - look backward from detected start
            search_start = max(0, detected_start - self.max_backward_search_frames)
            search_end = detected_start

            if search_start >= search_end:
                self.logger.debug(f"No backward search needed for {action_tag} at frame {detected_start}")
                return

            self.logger.debug(f"Refining start for {action_tag}: searching backward from {detected_start} to {search_start}")

            # Binary search for the exact start frame
            left, right = search_start, search_end
            refined_start = detected_start  # Default to detected start

            while left <= right:
                mid = (left + right) // 2

                async with vlm_semaphore:
                    try:
                        # Check VLM cache first
                        vlm_cache_key = (video_path, mid)
                        if vlm_cache_key in vlm_cache:
                            action_results = vlm_cache[vlm_cache_key]
                        else:
                            # Extract and analyze frame
                            with self._temp_frame(frame_extractor, video_path, mid, 'Phase 1.5') as (frame_tensor, frame_pil):
                                if frame_pil is None:
                                    break

                            action_results = await vlm_analyze_function(frame_pil)
                            video_metadata["api_calls_made"] += 1
                            self._cache_vlm_result(vlm_cache, vlm_cache_key, action_results)

                        confidence = action_results.get(action_tag, 0.0)
                        is_present = confidence >= threshold

                        if is_present:
                            # Action is present at this frame, search earlier
                            refined_start = mid
                            right = mid - 1
                        else:
                            # Action is absent, search later
                            left = mid + 1

                    except Exception as e:
                        self.logger.error(f"Error refining start for {action_tag} at frame {mid}: {e}")
                        break

            # Update segment with refined start
            if refined_start != detected_start:
                self.logger.debug(f"Refined start for {action_tag}: {detected_start} -> {refined_start}")
                segment["start_frame"] = refined_start
                segment["refined"] = True
            else:
                segment["refined"] = False

        # Refine all segments concurrently
        refinement_tasks = [refine_segment_start(segment) for segment in candidate_segments]
        await asyncio.gather(*refinement_tasks)

    @contextmanager
    def _temp_frame(self, frame_extractor, video_path, frame_idx, phase: str = ''):
        """Temporary frame extraction with cleanup"""
        frame_tensor = frame_extractor.extract_frame(video_path, frame_idx)
        if frame_tensor is None:
            yield None, None
            return
        frame_pil = self._convert_tensor_to_pil(frame_tensor)
        yield frame_tensor, frame_pil
        del frame_tensor, frame_pil
        gc.collect()

    def _convert_tensor_to_pil(self, frame_tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image"""
        if frame_tensor.dtype == torch.float16:
            frame_tensor = frame_tensor.float()

        # Ensure tensor is in [0, 1] range
        if frame_tensor.max() <= 1.0:
            frame_tensor = frame_tensor * 255.0

        frame_tensor = frame_tensor.clamp(0, 255).byte()

        # Convert to numpy and PIL
        frame_np = frame_tensor.cpu().numpy()
        if frame_np.ndim == 3 and frame_np.shape[0] == 3:
            frame_np = frame_np.transpose(1, 2, 0)

        return Image.fromarray(frame_np)

    def _cache_vlm_result(self, vlm_cache: Dict, cache_key: Tuple, action_results: Dict[str, float]):
        """Cache VLM analysis result with size limit"""
        vlm_cache[cache_key] = action_results

        # Implement LRU-style cache eviction if needed
        if len(vlm_cache) > 200:  # Cache size limit
            # Remove oldest entries
            oldest_keys = list(vlm_cache.keys())[:50]
            for key in oldest_keys:
                del vlm_cache[key]


class EndDeterminationStage(BasePipelineStage):
    """
    Stage 4: Parallel binary search to refine action ends (Phase 2)

    Inputs: video_metadata (with refined candidate_segments), vlm_analyze_function
    Outputs: video_metadata (with complete action ranges)
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    async def process_item(self, item: QueueItem) -> None:
        """Determine end boundaries for candidate segments using binary search"""
        item_future = item.item_future

        # Extract inputs
        video_metadata = item_future[item.input_names[0]]
        vlm_analyze_function = item_future[item.input_names[1]]

        candidate_segments = video_metadata.get("candidate_segments", [])

        if not candidate_segments:
            self.logger.info("No candidate segments for end determination, skipping Phase 2")
            await item_future.set_data(item.output_names[0], video_metadata)
            return

        self.logger.info(f"Phase 2: Determining end boundaries for {len(candidate_segments)} candidates")

        try:
            await self._phase2_binary_search(video_metadata, vlm_analyze_function, candidate_segments)

            self.logger.info(f"Phase 2 complete: Determined end boundaries for {len(candidate_segments)} segments")

            # Output complete metadata
            await item_future.set_data(item.output_names[0], video_metadata)

        except Exception as e:
            self.logger.error(f"Failed in end determination stage: {e}")
            raise

    async def _phase2_binary_search(
        self,
        video_metadata: Dict,
        vlm_analyze_function,
        candidate_segments: List[Dict[str, Any]]
    ) -> None:
        """
        Phase 2: Parallel binary search to refine action ends.

        For each candidate segment, perform binary search to find the exact end frame.
        """
        video_path = video_metadata["video_path"]
        total_frames = video_metadata["total_frames"]
        fps = video_metadata["fps"]
        use_timestamps = video_metadata["use_timestamps"]
        threshold = video_metadata["threshold"]
        vlm_semaphore = video_metadata["vlm_semaphore"]
        frame_extractor = video_metadata["frame_extractor"]
        vlm_cache = video_metadata["vlm_cache"]
        processed_frame_data = video_metadata["processed_frame_data"]

        # Initialize action ranges for binary search
        action_ranges = []
        for segment in candidate_segments:
            action_range = ActionRange(
                start_frame=segment["start_frame"],
                end_frame=total_frames - 1,
                action_tag=segment["action_tag"]
            )
            # Mark as searching for end boundary
            action_range.searching_end = True
            action_range.end_search_start = segment["start_frame"]
            action_range.end_search_end = total_frames - 1
            action_range.start_found = segment["start_frame"]
            action_ranges.append(action_range)

        # Initialize boundary detector and midpoint collector
        boundary_detector = ActionBoundaryDetector(threshold)
        midpoint_collector = AdaptiveMidpointCollector()

        iteration = 0
        max_iterations = 50  # Safety limit

        while self._has_unresolved_actions(action_ranges) and iteration < max_iterations:
            iteration += 1

            # Collect midpoints for binary search
            midpoints = midpoint_collector.collect_unique_midpoints(action_ranges)

            if not midpoints:
                self.logger.debug("No midpoints to process, ending Phase 2")
                break

            # Process midpoint frames concurrently
            async def process_midpoint_frame(frame_idx: int) -> Optional[Dict[str, Any]]:
                """Process a single frame in the binary search"""
                async with vlm_semaphore:
                    try:
                        # Check VLM cache first
                        vlm_cache_key = (video_path, frame_idx)
                        if vlm_cache_key in vlm_cache:
                            action_results = vlm_cache[vlm_cache_key]
                            self.logger.debug(f"VLM cache hit for frame {frame_idx}")
                        else:
                            # Extract frame
                            with self._temp_frame(frame_extractor, video_path, frame_idx, 'Phase 2') as (frame_tensor, frame_pil):
                                if frame_pil is None:
                                    return None

                            # Analyze frame with VLM
                            action_results = await vlm_analyze_function(frame_pil)
                            video_metadata["api_calls_made"] += 1

                            # Cache the VLM analysis result
                            self._cache_vlm_result(vlm_cache, vlm_cache_key, action_results)

                        return {
                            "frame_idx": frame_idx,
                            "action_results": action_results,
                            "timestamp": frame_idx / fps if use_timestamps else frame_idx
                        }

                    except Exception as e:
                        self.logger.error(f"Error processing frame {frame_idx} in Phase 2: {e}")
                        return None

            # Process all midpoint frames concurrently
            frame_tasks = [process_midpoint_frame(frame_idx) for frame_idx in midpoints]
            concurrent_results = await asyncio.gather(*frame_tasks, return_exceptions=True)

            # Process results and update boundaries
            for result in concurrent_results:
                if isinstance(result, Exception) or result is None:
                    continue

                frame_idx = result["frame_idx"]
                action_results = result["action_results"]

                # Update action boundaries based on this frame's results
                boundary_detector.update_action_boundaries(
                    action_ranges, frame_idx, action_results, total_frames
                )

                # Store frame result
                processed_frame_data[frame_idx] = result

        # Update candidate segments with end boundaries
        for i, action_range in enumerate(action_ranges):
            segment = candidate_segments[i]
            if action_range.end_found is not None:
                segment["end_frame"] = action_range.end_found
            else:
                # If no end found, set end to start (single frame action)
                segment["end_frame"] = segment["start_frame"]

            segment["resolved"] = action_range.is_resolved()

        self.logger.info(f"Phase 2 complete: Processed {len(processed_frame_data)} frames in {iteration} iterations")

    def _has_unresolved_actions(self, action_ranges: List[ActionRange]) -> bool:
        """Check if there are still actions being searched"""
        return any(not action_range.is_resolved() for action_range in action_ranges)

    @contextmanager
    def _temp_frame(self, frame_extractor, video_path, frame_idx, phase: str = ''):
        """Temporary frame extraction with cleanup"""
        frame_tensor = frame_extractor.extract_frame(video_path, frame_idx)
        if frame_tensor is None:
            yield None, None
            return
        frame_pil = self._convert_tensor_to_pil(frame_tensor)
        yield frame_tensor, frame_pil
        del frame_tensor, frame_pil
        gc.collect()

    def _convert_tensor_to_pil(self, frame_tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image"""
        if frame_tensor.dtype == torch.float16:
            frame_tensor = frame_tensor.float()

        # Ensure tensor is in [0, 1] range
        if frame_tensor.max() <= 1.0:
            frame_tensor = frame_tensor * 255.0

        frame_tensor = frame_tensor.clamp(0, 255).byte()

        # Convert to numpy and PIL
        frame_np = frame_tensor.cpu().numpy()
        if frame_np.ndim == 3 and frame_np.shape[0] == 3:
            frame_np = frame_np.transpose(1, 2, 0)

        return Image.fromarray(frame_np)

    def _cache_vlm_result(self, vlm_cache: Dict, cache_key: Tuple, action_results: Dict[str, float]):
        """Cache VLM analysis result with size limit"""
        vlm_cache[cache_key] = action_results

        # Implement LRU-style cache eviction if needed
        if len(vlm_cache) > 200:  # Cache size limit
            # Remove oldest entries
            oldest_keys = list(vlm_cache.keys())[:50]
            for key in oldest_keys:
                del vlm_cache[key]


class ResultCompilationStage(BasePipelineStage):
    """
    Stage 5: Aggregate results into final format compatible with existing postprocessing

    Inputs: video_metadata (with complete action ranges)
    Outputs: frame_results (List[Dict[str, Any]] compatible with existing postprocessing)
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    async def process_item(self, item: QueueItem) -> None:
        """Compile final results from all processed data"""
        item_future = item.item_future

        # Extract inputs
        video_metadata = item_future[item.input_names[0]]

        try:
            frame_results = self._compile_frame_results(video_metadata)

            self.logger.info(f"Result compilation complete: {len(frame_results)} frame results")

            # Output final results
            await item_future.set_data(item.output_names[0], frame_results)

        except Exception as e:
            self.logger.error(f"Failed in result compilation stage: {e}")
            raise

    def _compile_frame_results(self, video_metadata: Dict) -> List[Dict[str, Any]]:
        """
        Compile frame results from processed data and candidate segments.

        Returns results compatible with existing postprocessing pipeline.
        """
        processed_frame_data = video_metadata["processed_frame_data"]
        candidate_segments = video_metadata.get("candidate_segments", [])
        fps = video_metadata["fps"]
        use_timestamps = video_metadata["use_timestamps"]

        # Start with all processed frames
        frame_results = []

        # Add all frames that were processed during the pipeline
        for frame_idx, frame_data in processed_frame_data.items():
            frame_result = {
                "frame_index": frame_idx,
                "action_results": frame_data["action_results"],
                "timestamp": frame_idx / fps if use_timestamps else frame_idx
            }
            frame_results.append(frame_result)

        # Add segment boundary frames if not already included
        for segment in candidate_segments:
            start_frame = segment["start_frame"]
            end_frame = segment.get("end_frame", start_frame)
            action_tag = segment["action_tag"]

            # Ensure start frame is included
            if start_frame not in processed_frame_data:
                frame_result = {
                    "frame_index": start_frame,
                    "action_results": {action_tag: 1.0},  # High confidence for detected boundary
                    "timestamp": start_frame / fps if use_timestamps else start_frame
                }
                frame_results.append(frame_result)

            # Ensure end frame is included (if different from start)
            if end_frame != start_frame and end_frame not in processed_frame_data:
                frame_result = {
                    "frame_index": end_frame,
                    "action_results": {action_tag: 0.0},  # Low confidence for end boundary
                    "timestamp": end_frame / fps if use_timestamps else end_frame
                }
                frame_results.append(frame_result)

        # Sort results by frame index
        frame_results.sort(key=lambda x: x["frame_index"])

        # Log compilation statistics
        total_api_calls = video_metadata.get("api_calls_made", 0)
        total_frames = video_metadata["total_frames"]
        reduction_percentage = (1 - total_api_calls / (total_frames / video_metadata.get("frame_interval", 1.0))) * 100

        self.logger.info(f"Binary search completed:")
        self.logger.info(f"  - Total API calls: {total_api_calls}")
        self.logger.info(f"  - Total frames: {total_frames}")
        self.logger.info(f"  - API call reduction: {reduction_percentage:.1f}%")
        self.logger.info(f"  - Candidate segments found: {len(candidate_segments)}")
        self.logger.info(f"  - Frame results compiled: {len(frame_results)}")

        return frame_results
