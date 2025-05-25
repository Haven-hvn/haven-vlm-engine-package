import logging
import torch
from .base_model import Model # Updated import path
from .python_model import PythonModel as AiPythonModel # Updated import path
import time
from typing import Dict, Any, List, Optional, Union, Tuple, TextIO
from ..async_processing.queue_item import QueueItem, ItemFuture # Updated import path
import os # For path operations

# Placeholder for ModelRunner if ai_processing is not available/inspectable
# If ModelRunner is a known class, replace Any with it.
ModelRunner = Any

class AIModel(Model):
    def __init__(self, configValues: Dict[str, Any]):
        super().__init__(configValues) # Changed from Model.__init__ to super()
        self.max_model_batch_size: int = int(configValues.get("max_model_batch_size", 12))
        self.batch_size_per_VRAM_GB: Optional[float] = configValues.get("batch_size_per_VRAM_GB")
        self.model_file_name: Optional[str] = configValues.get("model_file_name") # User provides path or identifier
        self.model_license_name: Optional[str] = configValues.get("model_license_name") # User provides path or identifier
        self.model_threshold: Optional[float] = configValues.get("model_threshold")
        self.model_return_tags: bool = bool(configValues.get("model_return_tags", False))
        self.model_return_confidence: bool = bool(configValues.get("model_return_confidence", False))
        self.device: Optional[str] = configValues.get("device")
        self.fill_to_batch: bool = bool(configValues.get("fill_to_batch_size", True))
        self.model_image_size: Optional[Union[int, Tuple[int, int]]] = configValues.get("model_image_size")
        self.model_category: Optional[Union[str, List[str]]] = configValues.get("model_category")
        self.model_version: Optional[str] = str(configValues.get("model_version")) if configValues.get("model_version") is not None else None
        self.model_identifier: Optional[str] = configValues.get("model_identifier")
        self.category_mappings: Optional[Dict[int, int]] = configValues.get("category_mappings")
        self.normalization_config: Union[int, Dict[str, List[float]]] = configValues.get("normalization_config", 1)
        self.tags_file_path: Optional[str] = configValues.get("tags_file_path") # New config for tags path

        if self.model_file_name is None:
            # For VLM models, model_file_name might be an ID, not a file.
            # This check might need to be conditional based on model type if AIModel is also base for VLM.
            # For now, assuming local AI models always need a model_file_name.
            if not isinstance(self, VLMAIModel): # Check if it's not a VLMAIModel instance (requires VLMAIModel import)
                 raise ValueError("model_file_name is required for local AI models")

        if self.model_category is not None and isinstance(self.model_category, list) and len(self.model_category) > 1:
            if self.category_mappings is None:
                raise ValueError("category_mappings is required for models with more than one category")

        self.model: Optional[Union[AiPythonModel, ModelRunner]] = None
        self.tags: Dict[int, str] = {}

        if self.device is None:
            self.localdevice: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.localdevice: torch.device = torch.device(self.device)

        self.update_batch_with_mutli_models(1)

    def update_batch_with_mutli_models(self, model_count: int) -> None:
        batch_multipliers: List[float] = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        effective_model_count: int = min(model_count, len(batch_multipliers))
        if effective_model_count == 0: effective_model_count = 1

        if self.batch_size_per_VRAM_GB is not None and torch.cuda.is_available():
            try:
                multiplier: float = batch_multipliers[effective_model_count - 1]
                batch_size_temp: float = self.batch_size_per_VRAM_GB * multiplier
                gpuMemory: float = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                scaledBatchSize: int = custom_round(batch_size_temp * gpuMemory)
                self.max_model_batch_size = max(1, scaledBatchSize)
                self.max_batch_size = max(1, scaledBatchSize)
                self.max_queue_size = max(1, scaledBatchSize)
                self.logger.debug(f"Setting batch size to {self.max_model_batch_size} based on VRAM size of {gpuMemory:.2f} GB for model {self.model_identifier or self.model_file_name} ({model_count} models active)")
            except Exception as e_vram:
                self.logger.error(f"Could not set batch size based on VRAM for {self.model_identifier or self.model_file_name}: {e_vram}. Using default {self.max_model_batch_size}.")

    async def worker_function(self, data: List[QueueItem]) -> None:
        if not data: return
        try:
            first_image_tensor: torch.Tensor = data[0].item_future[data[0].input_names[0]]
            first_image_shape: torch.Size = first_image_tensor.shape
            images: torch.Tensor = torch.empty((len(data), *first_image_shape), dtype=first_image_tensor.dtype, device=self.localdevice)

            for i, item in enumerate(data):
                images[i] = item.item_future[item.input_names[0]]

            curr_time: float = time.time()
            if self.model is None or not hasattr(self.model, 'process_images'):
                 raise RuntimeError(f"Model {self.model_identifier or self.model_file_name} is not loaded or does not have process_images method.")

            model_results: Any = self.model.process_images(images)
            self.logger.debug(f"Processed {len(images)} images in {time.time() - curr_time:.3f}s in {self.model_identifier or self.model_file_name}")

            for i, item in enumerate(data):
                item_future: ItemFuture = item.item_future
                threshold: Optional[float] = item_future.get(item.input_names[1], self.model_threshold)
                return_confidence_override: Optional[bool] = item_future.get(item.input_names[2])
                current_return_confidence: bool = return_confidence_override if return_confidence_override is not None else self.model_return_confidence

                output_names_list: List[str] = [item.output_names] if isinstance(item.output_names, str) else item.output_names
                toReturn: Dict[str, List[Union[Tuple[str, float], str]]] = {output_name: [] for output_name in output_names_list}
                single_item_result: Any = model_results[i]

                if not self.tags:
                    self.logger.warning(f"Tags not loaded for model {self.model_identifier or self.model_file_name}. Skipping tag processing.")
                    # Set empty results for all output names to ensure future completion
                    for output_name_key in toReturn:
                        await item_future.set_data(output_name_key, [])
                    continue

                for j, confidence_value in enumerate(single_item_result):
                    tag_name: str = self.tags.get(j, f"unknown_tag_index_{j}")
                    processed_tag: Optional[Union[Tuple[str, float], str]] = None # Initialize to None

                    current_threshold: Optional[float] = float(threshold) if threshold is not None else None

                    if current_threshold is not None and confidence_value.item() > current_threshold:
                        if current_return_confidence:
                            processed_tag = (tag_name, round(confidence_value.item(), 2))
                        else:
                            processed_tag = tag_name
                    elif current_threshold is None:
                        if self.model_return_tags:
                            if current_return_confidence:
                                processed_tag = (tag_name, round(confidence_value.item(), 2))
                            else:
                                processed_tag = tag_name
                    
                    if processed_tag is None: # If tag wasn't set (e.g. threshold not met, or not returning tags)
                        continue

                    if self.category_mappings and j in self.category_mappings:
                        list_id_index: int = self.category_mappings[j]
                        if 0 <= list_id_index < len(output_names_list):
                            toReturn[output_names_list[list_id_index]].append(processed_tag)
                        else:
                            self.logger.warning(f"Category mapping index {list_id_index} out of bounds for output names.")
                    elif not self.category_mappings and output_names_list:
                        toReturn[output_names_list[0]].append(processed_tag)
                
                for output_name_key, result_val_list in toReturn.items():
                    await item_future.set_data(output_name_key, result_val_list)
        except Exception as e:
            self.logger.error(f"Error in AI model ({self.model_identifier or self.model_file_name}) worker_function: {e}", exc_info=True)
            for err_item in data:
                if hasattr(err_item, 'item_future') and err_item.item_future and not err_item.item_future.done():
                    err_item.item_future.set_exception(e)

    async def load(self) -> None:
        if self.model is None:
            # model_file_name is now expected to be a full path or an identifier the user manages
            if not self.model_file_name or not os.path.exists(self.model_file_name):
                 # This check might be too strict if model_file_name is an ID for a model registry
                 # For now, assume it's a file path for local .pt models
                if not isinstance(self, VLMAIModel): # Check if it's not a VLMAIModel instance
                    self.logger.error(f"Model file not found at {self.model_file_name}")
                    raise FileNotFoundError(f"Model file not found at {self.model_file_name}")

            self.logger.info(f"Loading model {self.model_identifier or self.model_file_name} with batch size {self.max_model_batch_size}, queue {self.max_queue_size}, batch {self.max_batch_size}")
            
            # Tags path is now explicitly provided in config via self.tags_file_path
            if not self.tags_file_path or not os.path.exists(self.tags_file_path):
                self.logger.warning(f"Tags file not found at {self.tags_file_path}. Tags will be empty.")
                self.tags = {}
            else:
                self.tags = get_index_to_tag_mapping(self.tags_file_path)

            try:
                if self.model_license_name is None:
                    # .pt model using AiPythonModel
                    # Ensure self.model_file_name is the actual path to the .pt file
                    self.model = AiPythonModel(self.model_file_name, self.max_model_batch_size, self.device, self.fill_to_batch)
                else:
                    # .pt.enc model using ModelRunner
                    # Ensure self.model_license_name is the actual path to the .lic file
                    if not os.path.exists(self.model_license_name):
                        self.logger.error(f"License file not found at {self.model_license_name}")
                        raise FileNotFoundError(f"License file not found at {self.model_license_name}")
                    try:
                        from ai_processing import ModelRunner as ExternalModelRunner # External, user-provided module
                        self.model = ExternalModelRunner(self.model_file_name, self.model_license_name, self.max_model_batch_size, self.device)
                    except ImportError:
                        self.logger.error("Module 'ai_processing' for ModelRunner not found. Encrypted models cannot be loaded.")
                        raise RuntimeError("ai_processing module not found for encrypted model.")
                
                # Default category mapping if single category and no explicit mapping
                if self.model_category and isinstance(self.model_category, str) and not self.category_mappings and self.tags:
                    self.category_mappings = {i: 0 for i in range(len(self.tags))}
                elif self.model_category and isinstance(self.model_category, list) and len(self.model_category) == 1 and not self.category_mappings and self.tags:
                     self.category_mappings = {i: 0 for i in range(len(self.tags))}

            except FileNotFoundError as fnf_e: # Should be caught by earlier checks ideally
                self.logger.error(f"Failed to load model {self.model_identifier or self.model_file_name}: {fnf_e}")
                raise
            except Exception as load_e:
                self.logger.error(f"Exception loading model {self.model_identifier or self.model_file_name}: {load_e}", exc_info=True)
                raise
        elif hasattr(self.model, 'load_model') and callable(self.model.load_model):
            await self.model.load_model() # type: ignore
        self.logger.info(f"Model {self.model_identifier or self.model_file_name} load process complete.")

def get_index_to_tag_mapping(path: str) -> Dict[int, str]:
    index_to_tag: Dict[int, str] = {}
    try:
        with open(path, 'r', encoding='utf-8') as file_handle:
            for index, line in enumerate(file_handle):
                index_to_tag[index] = line.strip()
    except FileNotFoundError:
        # Logger is instance specific, so can't use self.logger here.
        # Package users should ensure paths are correct.
        # Consider raising the error or returning empty and letting caller handle.
        logging.warning(f"Tags file not found at {path}. Tags will be empty.")
    return index_to_tag

def custom_round(value: float) -> int:
    if value < 8:
        return int(value)
    remainder: int = int(value) % 8
    if remainder <= 5:
        return int(value) - remainder
    else:
        return int(value) + (8 - remainder)

# Need to import VLMAIModel here to resolve the isinstance check,
# but this creates a circular dependency if VLMAIModel also imports AIModel.
# This suggests VLMAIModel should not inherit AIModel if this check is needed,
# or the check should be done differently (e.g. checking for a specific attribute).
# For now, I'll add a forward reference.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .vlm_ai_model import VLMAIModel
else:
    VLMAIModel = Any # Placeholder if not type checking
