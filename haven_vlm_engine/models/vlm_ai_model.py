import logging
import time
from PIL import Image
import numpy as np
from .ai_model import AIModel # Updated import
from .vlm_client import OpenAICompatibleVLMClient # Updated import
from typing import Dict, Any, List, Optional, Union, Tuple
from ..async_processing.queue_item import ItemFuture, QueueItem # Updated import

class VLMAIModel(AIModel):
    def __init__(self, configValues: Dict[str, Any]):
        # AIModel's __init__ expects "model_file_name".
        # For VLMs, this isn't a file but an identifier for the remote model.
        # We can use "model_id" from config for this purpose if "model_file_name" isn't explicitly set.
        if "model_file_name" not in configValues and "model_id" in configValues:
            configValues["model_file_name"] = str(configValues["model_id"]) # Ensure it's a string
        elif "model_file_name" not in configValues:
            # Fallback if neither is provided, though AIModel might raise error later if it expects a file path
            configValues["model_file_name"] = "remote_vlm_api" 
            
        super().__init__(configValues)
        
        # client_config is essentially configValues, OpenAICompatibleVLMClient will pick what it needs.
        self.client_config: Dict[str, Any] = configValues 

        # These are already set by AIModel's __init__, but can be overridden if VLM has different defaults/logic
        # self.max_model_batch_size = int(configValues.get("max_model_batch_size", 1)) # VLM typically batch 1
        # self.model_threshold = float(configValues.get("model_threshold", 0.5))
        # self.model_return_confidence = bool(configValues.get("model_return_confidence", True))
        # self.model_category = configValues.get("model_category")
        # self.model_version = str(configValues.get("model_version", "1.0"))
        # self.model_identifier = str(configValues.get("model_identifier", "default_vlm_identifier"))

        if self.model_category is None: # This check is also in AIModel, but good to ensure for VLM context
            raise ValueError("model_category is required for VLM AI models")

        # self.logger is inherited from AIModel
        self.vlm_model: Optional[OpenAICompatibleVLMClient] = None
        # self.localdevice is inherited from AIModel but might not be used if inputs are PIL/numpy

    async def worker_function(self, data: List[QueueItem]): # data will typically be a list of 1 item for VLMs
        if not data:
            return
        
        # VLMs usually process one image at a time.
        # The inherited ModelProcessor batching might send a list, but we'll iterate.
        # However, VLMAIModel's max_batch_size is often 1.
        item: QueueItem
        for item_idx, item in enumerate(data): # item_idx useful if batch > 1, though unlikely for VLM
            itemFuture: ItemFuture = item.item_future
            try:
                # Input name for image data, threshold, return_confidence should be defined in pipeline config
                image_input_name = item.input_names[0]
                threshold_input_name = item.input_names[1] if len(item.input_names) > 1 else None
                return_confidence_input_name = item.input_names[2] if len(item.input_names) > 2 else None

                image_data: Any = itemFuture[image_input_name]
                
                threshold_val: Optional[float] = itemFuture.get(threshold_input_name, self.model_threshold) if threshold_input_name else self.model_threshold
                threshold: float = float(threshold_val) if threshold_val is not None else self.model_threshold

                return_confidence_val: Optional[bool] = itemFuture.get(return_confidence_input_name, self.model_return_confidence) if return_confidence_input_name else self.model_return_confidence
                current_return_confidence: bool = bool(return_confidence_val) if return_confidence_val is not None else self.model_return_confidence

                image_pil: Image.Image
                if isinstance(image_data, Image.Image):
                    image_pil = image_data
                elif isinstance(image_data, np.ndarray):
                    # Convert numpy to PIL
                    if image_data.ndim == 3 and image_data.shape[0] == 3:  # CHW (common for PyTorch tensors)
                        image_data = np.transpose(image_data, (1, 2, 0)) # HWC
                    # Ensure data is uint8 for PIL
                    if image_data.dtype != np.uint8:
                        if image_data.dtype in [np.float32, np.float64] and image_data.min() >= 0 and image_data.max() <= 1:
                            image_data = (image_data * 255).astype(np.uint8)
                        else: # Attempt direct conversion, might fail if not suitable range/type
                            image_data = image_data.astype(np.uint8)
                    image_pil = Image.fromarray(image_data)
                elif hasattr(image_data, 'cpu') and hasattr(image_data, 'numpy'): # PyTorch tensor
                    image_np_tensor = image_data.cpu().numpy()
                    if image_np_tensor.ndim == 3 and image_np_tensor.shape[0] == 3:  # CHW
                        image_np_tensor = np.transpose(image_np_tensor, (1, 2, 0)) # HWC
                    if image_np_tensor.dtype != np.uint8:
                         if image_np_tensor.dtype in [np.float32, np.float64] and image_np_tensor.min() >= 0 and image_np_tensor.max() <= 1:
                            image_np_tensor = (image_np_tensor * 255).astype(np.uint8)
                         else:
                            image_np_tensor = image_np_tensor.astype(np.uint8)
                    image_pil = Image.fromarray(image_np_tensor)
                else:
                    self.logger.error(f"Unsupported image_data type: {type(image_data)} for VLM model {self.model_identifier}")
                    await itemFuture.set_exception(TypeError(f"Unsupported image_data type: {type(image_data)}"))
                    continue # Next item in batch if any

                if self.vlm_model is None:
                    self.logger.error(f"VLM model {self.model_identifier} not loaded.")
                    await itemFuture.set_exception(RuntimeError(f"VLM model {self.model_identifier} not loaded."))
                    continue

                start_time: float = time.time()
                scores: Dict[str, float] = self.vlm_model.analyze_frame(image_pil)
                self.logger.debug(f"VLM Model ({self.model_identifier}) processed frame in {time.time() - start_time:.3f}s. Scores: {scores}")
                
                output_names_list: List[str] = [item.output_names] if isinstance(item.output_names, str) else item.output_names
                toReturn: Dict[str, List[Union[Tuple[str, float], str]]] = {output_name: [] for output_name in output_names_list}
                
                for tag_name, score_confidence in scores.items():
                    if score_confidence > threshold:
                        processed_tag: Union[Tuple[str, float], str]
                        if current_return_confidence:
                            processed_tag = (tag_name, round(score_confidence, 2))
                        else:
                            processed_tag = tag_name
                        
                        # Determine target output list based on model_category
                        # This assumes model_category is a string or the first element of a list of strings
                        # that matches one of the output_names defined in the pipeline for this model.
                        target_output_category: Optional[str] = None
                        if isinstance(self.model_category, str):
                            target_output_category = self.model_category
                        elif isinstance(self.model_category, list) and self.model_category:
                            target_output_category = self.model_category[0] # Use first category

                        if target_output_category and target_output_category in toReturn:
                            toReturn[target_output_category].append(processed_tag)
                        elif output_names_list: # Fallback to the first defined output name
                             toReturn[output_names_list[0]].append(processed_tag)
                        # else: no valid output list to append to, tag is dropped.

                for output_name_key, result_list_val in toReturn.items():
                    await itemFuture.set_data(output_name_key, result_list_val)
            
            except Exception as e: # Catch exception per item
                self.logger.error(f"Error processing item with VLM AI model {self.model_identifier}: {e}", exc_info=True)
                if itemFuture and not itemFuture.done():
                    itemFuture.set_exception(e)
        
        # self.logger.debug(f"Batch of {len(data)} items processed with VLM AI model {self.model_identifier}")

    async def load(self) -> None:
        if self.vlm_model is None:
            self.logger.info(f"Loading VLM client for model_id: {self.client_config.get('model_id', self.model_identifier)}")
            try:
                self.vlm_model = OpenAICompatibleVLMClient(config=self.client_config)
                self.logger.info(f"OpenAICompatibleVLMClient for {self.client_config.get('model_id', self.model_identifier)} loaded successfully.")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAICompatibleVLMClient: {e}", exc_info=True)
                raise
        # No separate tags to load here as AIModel.load() handles its tags_file_path
        # and OpenAICompatibleVLMClient handles its own tag list internally from its config.
        # Call super().load() if AIModel's load has other essential base logic beyond tags.
        # For now, assuming AIModel.load() is primarily for its own .pt model and tags.
        # If VLMAIModel needs tags from AIModel's mechanism, that needs to be reconciled.
        # Based on current AIModel, its load() is for AiPythonModel or ModelRunner.
        # VLMAIModel does not use self.model (the AiPythonModel/ModelRunner instance), it uses self.vlm_model.
        # So, calling super().load() might try to load a non-existent local model.
        # Thus, we probably should NOT call super().load() here unless AIModel.load() is refactored.
        self.logger.info(f"VLM Model {self.model_identifier} load process complete (client initialized).")
