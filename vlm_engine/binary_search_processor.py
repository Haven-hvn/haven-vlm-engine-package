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
import asyncio
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
        # Process items concurrently instead of sequentially
        tasks = []
        for item in queue_items:
            tasks.append(asyncio.create_task(self._process_single_item(item)))
        
        # Wait for all items to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_single_item(self, item: QueueItem) -> None:
        """Process a single video item with binary search"""
        try:
            await self._process_video_item(item)
        except Exception as e:
            self.logger.error(f"Error processing video item: {e}", exc_info=True)
            if hasattr(item, 'item_future') and item.item_future:
                item.item_future.set_exception(e)
    
    async def _process_video_item(self, item: QueueItem) -> None:
        """Core video processing logic for a single item"""
        item_future: ItemFuture = item.item_future
        
        # Validate item_future has data
        if item_future.data is None:
            self.logger.error(f"ItemFuture.data is None - cannot process item. Input names: {item.input_names}")
            # Try to get some info about the item itself
            if hasattr(item, 'item_future') and item.item_future is not None:
                self.logger.error(f"ItemFuture attributes: {dir(item_future)}")
            raise ValueError("ItemFuture data is None - item may have been already processed or failed to initialize")
        
        video_path: str = item_future[item.input_names[0]]
        if video_path is None:
            self.logger.error(f"video_path is None. Input names: {item.input_names}")
            self.logger.error(f"ItemFuture.data keys: {list(item_future.data.keys()) if item_future.data else 'None'}")
            raise ValueError(f"video_path (key '{item.input_names[0]}') not found in ItemFuture.data")
        
        use_timestamps: bool = item_future[item.input_names[1]]
        frame_interval_override: Optional[float] = item_future[item.input_names[2]] if item.input_names[2] in item_future else None
        threshold: float = item_future[item.input_names[3]] if item.input_names[3] in item_future else 0.5
        return_confidence: bool = item_future[item.input_names[4]] if item.input_names[4] in item_future else True
        vr_video: bool = item_future[item.input_names[5]] if len(item.input_names) > 5 and item.input_names[5] in item_future else False
        existing_video_data: Optional[Dict] = item_future[item.input_names[6]] if len(item.input_names) > 6 and item.input_names[6] in item_future else None
        skipped_categories: Optional[List] = item_future[item.input_names[7]] if len(item.input_names) > 7 and item.input_names[7] in item_future else None

        self.logger.info(f"[DEBUG] BinarySearchProcessor._process_video_item: Starting processing for {video_path}")
        self.logger.info(f"[DEBUG] ItemFuture.data keys: {list(item_future.data.keys()) if item_future.data else 'None'}")
        self.logger.info(f"[DEBUG] Input names: {item.input_names}")
        self.logger.info(f"[DEBUG] Output names: {item.output_names}")

        callback = item_future["callback"] if "callback" in item_future else None
        if callback:
            callback(0)

        # Get VLM configuration from pipeline
        vlm_config = self._extract_vlm_config(item_future)
        self.logger.info(f"[DEBUG] VLM config extracted: {vlm_config is not None}")
        if vlm_config is None:
            self.logger.error("No VLM configuration found - falling back to linear processing")
            # Pass video_path since item_future.data might be cleared later
            await self._fallback_linear_processing(item, video_path, use_timestamps,
                                                frame_interval_override, threshold,
                                                return_confidence, vr_video)
            return

        # Extract action tags from VLM config
        action_tags = vlm_config.get("tag_list", [])
        self.logger.info(f"[DEBUG] Action tags found: {len(action_tags)} tags")
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
            use_half_precision=self.use_half_precision,
            progress_callback=callback
        )

        # Get VLM coordinator from pipeline
        vlm_coordinator = self._get_vlm_coordinator(item_future)
        self.logger.info(f"[DEBUG] VLM coordinator obtained: {vlm_coordinator is not None}")
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
            frame_interval=frame_interval_override if frame_interval_override is not None else 1.0,
            use_timestamps=use_timestamps,
            max_concurrent_vlm_calls=10  # Increase from default of 10 for more concurrent VLM requests per video
        )
        
        if callback:
            callback(90)

        # Sort frame results by frame_index to ensure chronological order for postprocessing
        # This is critical because binary search processes frames out of order, but the
        # postprocessing pipeline expects chronological order for proper timespan construction
        frame_results.sort(key=lambda x: x["frame_index"])

        # Convert frame results to children without post-processing
        # Post-processing (mutual exclusivity) will be handled in the postprocessing stage
        children = []
        for fr in frame_results:
            frame_index = fr["frame_index"]
            # Convert action_results to match model_category output format
            # The model_category is used as the key for VLM results
            humanactivityevaluation = []
            for action_tag, confidence in fr["action_results"].items():
                humanactivityevaluation.append((action_tag, confidence))

            self.logger.debug(f'Creating child for frame_index: {frame_index}, humanactivityevaluation: {humanactivityevaluation}')

            # Create child with empty data - let pipeline event handler control model execution
            # Only set frame_index and the detection results (matching model_category)
            result_future = await ItemFuture.create(item_future, {}, item_future.handler)
            await result_future.set_data("frame_index", frame_index)
            await result_future.set_data("humanactivityevaluation", humanactivityevaluation)
            self.logger.debug(f"[DEBUG] Child ItemFuture data keys after set_data: {list(result_future.data.keys())}")
            children.append(result_future)
        
        await item_future.set_data(item.output_names[0], children)
        
        # Set frame_results for video_result_postprocessor which expects this key
        await item_future.set_data("frame_results", frame_results)
        
        # Pass through any other parameters needed by downstream processors
        if video_path:
            await item_future.set_data("video_path", video_path)
        if frame_interval_override is not None:
            await item_future.set_data("time_interval", frame_interval_override)
        if threshold is not None:
            await item_future.set_data("threshold", threshold)
        if existing_video_data is not None:
            await item_future.set_data("existing_video_data", existing_video_data)
        
        self.logger.info(f"Binary search completed: {len(children)} frames processed with {engine.api_calls_made} API calls")
        self.logger.info(f"[DEBUG] Child ItemFuture data keys sample: {list(children[0].data.keys()) if children else 'N/A'}")

    def _extract_vlm_config(self, item_future: ItemFuture) -> Optional[Dict[str, Any]]:
        """Extract VLM configuration from pipeline context"""
        try:
            # Try to get pipeline configuration
            pipeline = None
            if hasattr(item_future, '__contains__') and "pipeline" in item_future:
                pipeline = item_future["pipeline"]
            elif hasattr(item_future, 'data') and item_future.data and "pipeline" in item_future.data:
                pipeline = item_future.data.get("pipeline")
            
            self.logger.debug(f"[DEBUG_VLM_CONFIG] Looking for pipeline in item_future: {'pipeline' in item_future if hasattr(item_future, '__contains__') else 'N/A'}")
            if pipeline:
                self.logger.debug(f"[DEBUG_VLM_CONFIG] Pipeline has {len(pipeline.models)} models")
                # Look for VLM model configuration
                for idx, model_wrapper in enumerate(pipeline.models):
                    self.logger.debug(f"[DEBUG_VLM_CONFIG] Model {idx}: name={model_wrapper.model_name_for_logging}, has model attr={hasattr(model_wrapper, 'model')}")
                    if hasattr(model_wrapper, 'model'):
                        self.logger.debug(f"[DEBUG_VLM_CONFIG] Model.model type: {type(model_wrapper.model).__name__}")
                        # Try multiple paths to get to the VLM config
                        vlm_config = None
                        
                        # Path 1: Check if model has client_config directly
                        if hasattr(model_wrapper.model, 'client_config'):
                            vlm_config = model_wrapper.model.client_config.model_dump()
                        
                        # Path 2: Check if model.model has client_config (nested model structure)
                        if not vlm_config and hasattr(model_wrapper.model, 'model'):
                            self.logger.debug(f"[DEBUG_VLM_CONFIG] Path 2: model_wrapper.model.model exists, type={type(model_wrapper.model.model).__name__}")
                            if hasattr(model_wrapper.model.model, 'client_config'):
                                self.logger.debug(f"[DEBUG_VLM_CONFIG] Path 2: model_wrapper.model.model has client_config")
                                vlm_config = model_wrapper.model.model.client_config.model_dump()
                            else:
                                self.logger.debug(f"[DEBUG_VLM_CONFIG] Path 2: model_wrapper.model.model does NOT have client_config, attrs={dir(model_wrapper.model.model)}")
                        
                        if vlm_config:
                            self.logger.debug(f"[DEBUG_VLM_CONFIG] Found VLM config with keys: {list(vlm_config.keys())}")
                            self.logger.debug(f"[DEBUG_VLM_CONFIG] tag_list: {vlm_config.get('tag_list', 'NOT FOUND')}")
                            # Ensure tag_list is a list
                            if 'tag_list' in vlm_config and vlm_config['tag_list']:
                                return vlm_config
                            else:
                                self.logger.warning(f"[DEBUG_VLM_CONFIG] VLM config found but tag_list is missing or empty: {vlm_config.get('tag_list')}")
            self.logger.warning("[DEBUG_VLM_CONFIG] No VLM configuration found in pipeline")
            return None
        except Exception as e:
            self.logger.error(f"Failed to extract VLM config: {e}", exc_info=True)
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

    async def _fallback_linear_processing(self, item: QueueItem, video_path: str,
                                     use_timestamps: bool, frame_interval_override: Optional[float],
                                     threshold: float, return_confidence: bool, vr_video: bool) -> None:
        """Fallback to original linear processing if binary search fails"""
        
        item_future = item.item_future
        self.logger.info(f"[DEBUG] FALLBACK: Using linear processing for video")

        current_frame_interval: float = frame_interval_override if frame_interval_override is not None else 0.5

        # Get callback but handle case where data might be None
        callback = None
        if item_future.data is not None and "callback" in item_future:
            callback = item_future["callback"]
        if callback:
            callback(0)

        from .preprocessing import get_video_duration_decord
        duration = get_video_duration_decord(video_path)
        expected_frames = int(duration / current_frame_interval) + 1

        children = []
        processed_frames_count = 0

        for frame_index, frame_tensor in preprocess_video(
            video_path, current_frame_interval, 512, self.use_half_precision,
            use_timestamps, vr_video=vr_video, norm_config_idx=1,
            process_for_vlm=self.process_for_vlm
        ):
            processed_frames_count += 1

            future_data_payload = {
                "dynamic_frame": frame_tensor,
                "frame_index": frame_index,
                "dynamic_threshold": threshold,
                "dynamic_return_confidence": return_confidence,
                "dynamic_skipped_categories": None
            }
            result_future = await ItemFuture.create(item_future, future_data_payload, item_future.handler)
            await result_future.set_data("frame_index", frame_index)
            children.append(result_future)

            if callback and processed_frames_count % 10 == 0:  # Update every 10 frames
                progress = int(90 * (processed_frames_count / expected_frames))
                callback(progress)

        await item_future.set_data(item.output_names[0], children)
        if callback:
            callback(100)
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
