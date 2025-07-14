"""
Replacement for VideoPreprocessorModel that uses parallel binary search.
Maintains complete external API compatibility.
"""

from .action_range import ActionRange
from .adaptive_midpoint_collector import AdaptiveMidpointCollector
from .action_boundary_detector import ActionBoundaryDetector
from .video_frame_extractor import VideoFrameExtractor
from .preprocessing import get_video_duration_decord, crop_black_bars_lr, is_macos_arm, preprocess_video
from PIL import Image
from collections import defaultdict
import logging
from typing import Any, Dict, List, Optional

from .async_utils import ItemFuture, QueueItem
from .config_models import ModelConfig
from .parallel_binary_search_engine import ParallelBinarySearchEngine
from .vlm_batch_coordinator import IntegratedVLMCoordinator

class BinarySearchProcessor:
    """
    Replacement for VideoPreprocessorModel that uses parallel binary search.
    Maintains complete external API compatibility.
    """

    def __init__(self, model_config: ModelConfig):
        self.logger = logging.getLogger("logger")
        self.device = model_config.device or "cpu"
        self.use_half_precision = True
        self.process_for_vlm = True  # Always enable VLM mode for binary search
        self.binary_search_enabled = True

        # Required attributes for ModelProcessor compatibility
        self.instance_count: int = model_config.instance_count
        self.max_queue_size: Optional[int] = model_config.max_queue_size
        self.max_batch_size: int = model_config.max_batch_size

        self.logger.info("BinarySearchProcessor initialized - parallel binary search enabled")

    def set_vlm_pipeline_mode(self, mode: bool) -> None:
        """Maintain compatibility with existing pipeline"""
        self.process_for_vlm = mode
        self.logger.info(f"BinarySearchProcessor VLM mode set to: {self.process_for_vlm}")

    async def worker_function(self, queue_items: List[QueueItem]) -> None:
        """Main processing function - replaces linear preprocessing with binary search"""
        for item in queue_items:
            item_future: ItemFuture = item.item_future
            try:
                video_path: str = item_future[item.input_names[0]]
                use_timestamps: bool = item_future[item.input_names[1]]
                frame_interval_override: Optional[float] = item_future[item.input_names[2]] if item.input_names[2] in item_future else None
                threshold: float = item_future[item.input_names[3]] if item.input_names[3] in item_future else 0.5
                return_confidence: bool = item_future[item.input_names[4]] if item.input_names[4] in item_future else True

                # Get VLM configuration from pipeline
                vlm_config = self._extract_vlm_config(item_future)
                if vlm_config is None:
                    self.logger.error("No VLM configuration found - falling back to linear processing")
                    await self._fallback_linear_processing(item)
                    return

                # Extract action tags from VLM config
                action_tags = vlm_config.get("tag_list", [])
                if not action_tags:
                    self.logger.error("No action tags found in VLM config")
                    await item_future.set_data(item.output_names[0], [])
                    return

                if not self.binary_search_enabled or not self.process_for_vlm:
                    self.logger.info("Binary search disabled or not in VLM mode - using linear processing")
                    await self._fallback_linear_processing(item)
                    return

                # Initialize binary search engine
                engine = ParallelBinarySearchEngine(
                    action_tags=action_tags,
                    threshold=threshold,
                    device_str=self.device,
                    use_half_precision=self.use_half_precision
                )

                # Get VLM coordinator from pipeline
                vlm_coordinator = self._get_vlm_coordinator(item_future)
                if vlm_coordinator is None:
                    self.logger.error("No VLM coordinator available - falling back to linear processing")
                    await self._fallback_linear_processing(item)
                    return

                # Create VLM analyzer function
                async def vlm_analyze_function(frame_pil: Image.Image) -> Dict[str, float]:
                    """Wrapper function for VLM analysis using actual VLM coordinator"""
                    assert vlm_coordinator is not None, "VLM coordinator must be initialized"
                    return await vlm_coordinator.analyze_frame(frame_pil)

                # Execute binary search
                frame_results = await engine.process_video_binary_search(
                    video_path=video_path,
                    vlm_analyze_function=vlm_analyze_function,
                    use_timestamps=use_timestamps
                )

                # Sort frame results by frame_index to ensure chronological order for postprocessing
                # This is critical because binary search processes frames out of order, but the
                # postprocessing pipeline expects chronological order for proper timespan construction
                frame_results.sort(key=lambda x: x["frame_index"])

                # Post-processing to enforce mutual exclusivity with prevalence priority
                current_frame_interval = frame_interval_override if frame_interval_override is not None else 0.5
                detected_segments = engine.get_detected_segments()
                duration = get_video_duration_decord(video_path)
                fps = engine.total_frames / duration if duration > 0 else 30.0
                # Group segments by tag
                tag_to_segments = defaultdict(list)
                for seg in detected_segments:
                    tag_to_segments[seg["action_tag"]].append((seg["start"], seg["end"]))
                # Calculate prevalence as occurrence count
                tag_to_prevalence = {tag: len(segs) for tag, segs in tag_to_segments.items()}
                # Sort tags by prevalence ascending (lower prevalence higher priority)
                priority_tags = sorted(tag_to_prevalence, key=lambda t: tag_to_prevalence[t])
                # Resolve overlaps
                occupied = []
                final_tag_to_segments = defaultdict(list)
                def subtract_from_interval(s, e, occ):
                    remaining = [(s, e)]
                    for o_s, o_e in sorted(occ):
                        new_rem = []
                        for r_s, r_e in remaining:
                            if r_e < o_s or r_s > o_e:
                                new_rem.append((r_s, r_e))
                            else:
                                if r_s < o_s:
                                    new_rem.append((r_s, o_s - 1))
                                if r_e > o_e:
                                    new_rem.append((o_e + 1, r_e))
                        remaining = new_rem
                    return [r for r in remaining if r[0] <= r[1]]
                def add_to_occupied(occ, s, e):
                    if s > e:
                        return
                    i = 0
                    start = s
                    end = e
                    while i < len(occ):
                        o_s, o_e = occ[i]
                        if o_e < start - 1:
                            i += 1
                            continue
                        if o_s > end + 1:
                            break
                        start = min(start, o_s)
                        end = max(end, o_e)
                        del occ[i]
                    occ.insert(i, (start, end))
                    occ.sort()
                for tag in priority_tags:
                    for s, e in sorted(tag_to_segments[tag]):
                        remaining = subtract_from_interval(s, e, occupied)
                        final_tag_to_segments[tag].extend(remaining)
                        for rs, re in remaining:
                            add_to_occupied(occupied, rs, re)
                # Generate all frame results
                frame_dict = {}
                # Add virtual frames
                for tag, segs in final_tag_to_segments.items():
                    for s, e in segs:
                        f = s
                        while f <= e:
                            frame_index = float(f) / fps if use_timestamps else int(f)
                            key = f
                            if key not in frame_dict:
                                frame_dict[key] = {"frame_index": frame_index, "actiondetection": [(tag, 1.0)]}
                            f += int(fps * current_frame_interval)
                # Adjust sampled frames
                for fr in frame_results:
                    f_idx = fr["frame_idx"]
                    frame_index = fr["frame_index"]
                    assigned_tag = None
                    conf = 1.0
                    for tag, segs in final_tag_to_segments.items():
                        if any(s <= f_idx <= e for s, e in segs):
                            assigned_tag = tag
                            conf = fr["action_results"].get(tag, 1.0)
                            break
                    actiondetection = [(assigned_tag, conf)] if assigned_tag else []
                    frame_dict[f_idx] = {"frame_index": frame_index, "actiondetection": actiondetection}
                all_frame_results = sorted(frame_dict.values(), key=lambda x: x["frame_index"])
                # Convert to children
                children = []
                for fr in all_frame_results:
                    frame_index = fr["frame_index"]
                    actiondetection = fr["actiondetection"]
                    self.logger.debug(f'Creating child for frame_index: {frame_index}, actiondetection: {actiondetection}')
                    result_future = await ItemFuture.create(item_future, {}, item_future.handler)
                    await result_future.set_data("frame_index", frame_index)
                    await result_future.set_data("actiondetection", actiondetection)
                    children.append(result_future)
                await item_future.set_data(item.output_names[0], children)
                self.logger.info(f"Binary search completed: {len(children)} frames processed with {engine.api_calls_made} API calls and mutual exclusivity enforced")

            except Exception as e:
                self.logger.error(f"BinarySearchProcessor error: {e}", exc_info=True)
                item_future.set_exception(e)

    def _extract_vlm_config(self, item_future: ItemFuture) -> Optional[Dict[str, Any]]:
        """Extract VLM configuration from pipeline context"""
        try:
            # Try to get pipeline configuration
            pipeline = item_future["pipeline"] if "pipeline" in item_future else None
            if pipeline:
                # Look for VLM model configuration
                for model_wrapper in pipeline.models:
                    if hasattr(model_wrapper.model, 'model') and hasattr(model_wrapper.model.model, 'client_config'):
                        return model_wrapper.model.model.client_config.dict()
            return None
        except Exception as e:
            self.logger.error(f"Failed to extract VLM config: {e}")
            return None

    def _get_vlm_coordinator(self, item_future: ItemFuture):
        """Get VLM coordinator from pipeline context"""
        try:
            pipeline = item_future["pipeline"] if "pipeline" in item_future else None
            if pipeline:
                # Create integrated VLM coordinator from pipeline models
                coordinator = IntegratedVLMCoordinator(pipeline.models)
                if coordinator.vlm_client is not None:
                    return coordinator

            self.logger.warning("No VLM coordinator could be created from pipeline")
            return None
        except Exception as e:
            self.logger.error(f"Failed to get VLM coordinator: {e}")
            return None

    async def _fallback_linear_processing(self, item: QueueItem) -> None:
        """Fallback to original linear processing if binary search fails"""
        
        item_future = item.item_future

        video_path: str = item_future[item.input_names[0]]
        use_timestamps: bool = item_future[item.input_names[1]]
        frame_interval_override: Optional[float] = item_future[item.input_names[2]] if item.input_names[2] in item_future else None
        current_frame_interval: float = frame_interval_override if frame_interval_override is not None else 0.5
        vr_video: bool = item_future[item.input_names[5]] if item.input_names[5] in item_future else False

        children = []
        processed_frames_count = 0

        for frame_index, frame_tensor in preprocess_video(
            video_path, current_frame_interval, 512, self.use_half_precision,
            self.device, use_timestamps, vr_video=vr_video, norm_config_idx=1,
            process_for_vlm=self.process_for_vlm
        ):
            processed_frames_count += 1

            future_data_payload = {
                "dynamic_frame": frame_tensor,
                "frame_index": frame_index,
                "dynamic_threshold": item_future[item.input_names[3]] if item.input_names[3] in item_future else 0.5,
                "dynamic_return_confidence": item_future[item.input_names[4]] if item.input_names[4] in item_future else True,
                "dynamic_skipped_categories": item_future[item.input_names[6]] if item.input_names[6] in item_future else None
            }
            result_future = await ItemFuture.create(item_future, future_data_payload, item_future.handler)
            await result_future.set_data("frame_index", frame_index)
            children.append(result_future)

        await item_future.set_data(item.output_names[0], children)
        self.logger.info(f"Fallback linear processing completed: {processed_frames_count} frames")

    async def load(self) -> None:
        """Required method for ModelProcessor compatibility"""
        self.logger.info("BinarySearchProcessor loaded successfully")

    async def worker_function_wrapper(self, data: List[QueueItem]) -> None:
        """Wrapper for worker_function to handle exceptions"""
        try:
            await self.worker_function(data)
        except Exception as e:
            self.logger.error(f"Exception in BinarySearchProcessor worker_function: {e}", exc_info=True)
            for item in data:
                if hasattr(item, 'item_future') and item.item_future:
                    item.item_future.set_exception(e)
