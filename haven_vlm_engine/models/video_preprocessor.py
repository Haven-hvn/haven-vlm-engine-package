import logging
import time
from ..async_processing.queue_item import ItemFuture, QueueItem # Updated import
from .base_model import Model # Updated import
from ..preprocessing.image_preprocessing import preprocess_video # Updated import
from typing import Dict, Any, List, Optional, Union, Tuple

class VideoPreprocessorModel(Model):
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        # self.logger is inherited from Model
        self.device: str = model_config.get("device", "cpu") # Consider if torch is available for "cuda"
        self.image_size: Union[int, List[int], Tuple[int, int]] = model_config.get("image_size", 512) # Allow Tuple
        self.frame_interval: float = float(model_config.get("frame_interval", 0.5))
        self.use_half_precision: bool = bool(model_config.get("use_half_precision", True))
        self.normalization_config: Union[int, Dict[str, List[float]]] = model_config.get("normalization_config", 1)
        self.process_for_vlm: bool = bool(model_config.get("process_for_vlm", False)) # Can be set in config
    
    def set_vlm_pipeline_mode(self, mode: bool) -> None:
        """Allows dynamic override if needed, though config is preferred."""
        self.process_for_vlm = mode
        self.logger.info(f"VideoPreprocessorModel VLM mode set to: {self.process_for_vlm}")

    async def worker_function(self, queue_items: List[QueueItem]) -> None:
        item: QueueItem
        for item in queue_items:
            itemFuture: ItemFuture = item.item_future
            try:
                totalTime: float = 0.0
                # Input names should be defined in the pipeline configuration for this model
                video_path_input_name = item.input_names[0] # e.g., "video_path"
                use_timestamps_input_name = item.input_names[1] # e.g., "return_timestamps"
                frame_interval_override_input_name = item.input_names[2] # e.g., "time_interval"
                # item.input_names[3] and [4] are often threshold and return_confidence for AI models,
                # but for preprocessor, they might be different or not used directly.
                # For vr_video, it was item.input_names[5] in original.
                vr_video_input_name = item.input_names[5] if len(item.input_names) > 5 else None


                video_path: str = itemFuture[video_path_input_name]
                use_timestamps: bool = bool(itemFuture.get(use_timestamps_input_name, False)) # Default to False if not provided
                
                frame_interval_override_val: Optional[float] = itemFuture.get(frame_interval_override_input_name)
                current_frame_interval: float = float(frame_interval_override_val) if frame_interval_override_val is not None else self.frame_interval
                
                vr_video: bool = bool(itemFuture.get(vr_video_input_name, False)) if vr_video_input_name else False

                children: List[ItemFuture] = []
                processed_frames_count: int = 0
                oldTime: float = time.time()
                
                # Output names should also be defined in pipeline config for this model
                # Original output_names: [children_futures_list, frame_tensor, frame_index, threshold, return_confidence, existing_video_data]
                # Let's assume output_names[0] is for the list of child futures (if generating multiple frames)
                # or for the single frame tensor if not batching frames from one video into multiple futures.
                # The original code created a list of futures, one per frame.
                # output_names[0] = "child_frame_futures" (list of futures)
                # For each child future, its data payload had:
                # item.output_names[1] -> frame_tensor
                # item.output_names[2] -> frame_index
                # item.output_names[3] -> (original) itemFuture[item.input_names[3]] (e.g. threshold)
                # item.output_names[4] -> (original) itemFuture[item.input_names[4]] (e.g. return_confidence)
                # item.output_names[5] -> (original) itemFuture[item.input_names[6]] (e.g. existing_video_data)

                child_output_map_frame_tensor = item.output_names[1] 
                child_output_map_frame_index = item.output_names[2]
                # Pass through other relevant inputs to children if needed
                # This requires careful pipeline design for what children expect.
                # Example: if an AI model later needs the original threshold.
                passthrough_inputs_map = {}
                if len(item.input_names) > 3 and len(item.output_names) > 3 and item.input_names[3] is not None:
                    passthrough_inputs_map[item.output_names[3]] = itemFuture.get(item.input_names[3])
                if len(item.input_names) > 4 and len(item.output_names) > 4 and item.input_names[4] is not None:
                    passthrough_inputs_map[item.output_names[4]] = itemFuture.get(item.input_names[4])
                if len(item.input_names) > 6 and len(item.output_names) > 5 and item.input_names[6] is not None: # original existing_video_data
                    passthrough_inputs_map[item.output_names[5]] = itemFuture.get(item.input_names[6])


                frame_index_val: int # Type hint for loop var
                frame_tensor_val: Any # Type hint for loop var
                for frame_index_val, frame_tensor_val in preprocess_video(
                    video_path, current_frame_interval, self.image_size, 
                    self.use_half_precision, self.device, use_timestamps, 
                    vr_video=vr_video, norm_config_idx=self.normalization_config, 
                    process_for_vlm=self.process_for_vlm
                ):
                    processed_frames_count += 1
                    newTime: float = time.time()
                    totalTime += newTime - oldTime
                    oldTime = newTime
                    
                    future_data_payload: Dict[str, Any] = {
                        child_output_map_frame_tensor: frame_tensor_val, 
                        child_output_map_frame_index: frame_index_val,
                        **passthrough_inputs_map # Add passthrough inputs
                    }
                    # Create a new ItemFuture for each processed frame
                    result_future: ItemFuture = await ItemFuture.create(item, future_data_payload, item.item_future.handler)
                    children.append(result_future)
                
                if processed_frames_count > 0:
                    avg_time_per_frame = totalTime / processed_frames_count if processed_frames_count > 0 else 0
                    self.logger.debug(f"Preprocessed {processed_frames_count} frames from {video_path} in {totalTime:.2f}s (avg: {avg_time_per_frame:.3f}s/frame).")
                else:
                    self.logger.debug(f"No frames preprocessed for {video_path}.")
                
                # The main output of this preprocessor model is the list of child futures
                await itemFuture.set_data(item.output_names[0], children)

            except FileNotFoundError as fnf_error:
                video_file_path_for_log = itemFuture.get(item.input_names[0], 'unknown_file')
                self.logger.error(f"File not found error processing {video_file_path_for_log}: {fnf_error}")
                if not itemFuture.done(): itemFuture.set_exception(fnf_error)
            except IOError as io_error: # Often indicates corrupted video
                video_file_path_for_log = itemFuture.get(item.input_names[0], 'unknown_file')
                self.logger.error(f"IO error (video might be corrupted) processing {video_file_path_for_log}: {io_error}")
                if not itemFuture.done(): itemFuture.set_exception(io_error)
            except Exception as e:
                video_file_path_for_log = itemFuture.get(item.input_names[0], 'unknown_file')
                self.logger.error(f"An unexpected error occurred processing {video_file_path_for_log}: {e}", exc_info=True)
                if not itemFuture.done(): itemFuture.set_exception(e)

    async def load(self) -> None:
        # VideoPreprocessorModel typically doesn't load a "model" file in the traditional sense.
        # Its "loading" is being ready to use torchvision/cv2 for preprocessing.
        # Dependencies like OpenCV or PyTorch should be handled by package installation.
        self.logger.info(f"VideoPreprocessorModel ({self.__class__.__name__}) is ready.")
        return
