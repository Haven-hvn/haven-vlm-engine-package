import logging
import time
from ..async_processing.queue_item import ItemFuture, QueueItem
from .base_model import Model
from ..preprocessing.image_preprocessing import preprocess_image_from_path # Renamed function
from typing import Dict, Any, List, Optional, Union, Tuple
from PIL import Image as PILImage
import torch

class ImagePreprocessorModel(Model):
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        # self.logger is inherited from Model
        self.device_str: Optional[str] = model_config.get("device") # Can be None, preprocess_image_from_path will default
        self.image_size: Union[int, Tuple[int,int]] = model_config.get("image_size", 512)
        self.use_half_precision: bool = bool(model_config.get("use_half_precision", True))
        self.normalization_config: Union[int, Dict[str, List[float]]] = model_config.get("normalization_config", 1)
        # Add any other image-specific preprocessing params from config if needed

    async def worker_function(self, queue_items: List[QueueItem]) -> None:
        item: QueueItem
        for item in queue_items:
            itemFuture: ItemFuture = item.item_future
            try:
                # Assume input_names[0] is the image path or PIL image or tensor
                image_path_or_data_input_name = item.input_names[0]
                image_path_or_data: Union[str, PILImage.Image, torch.Tensor] = itemFuture[image_path_or_data_input_name]

                start_time: float = time.time()
                
                processed_tensor: torch.Tensor = preprocess_image_from_path(
                    image_path_or_data,
                    img_size=self.image_size,
                    use_half_precision=self.use_half_precision,
                    device_str=self.device_str,
                    norm_config_idx=self.normalization_config
                )
                
                processing_time = time.time() - start_time
                self.logger.debug(f"Preprocessed image in {processing_time:.3f} seconds.")

                # Assume output_names[0] is for the processed tensor
                output_tensor_name = item.output_names[0]
                await itemFuture.set_data(output_tensor_name, processed_tensor)

                # If there are other outputs to pass through (e.g., metadata, original path)
                # This needs to be defined in the pipeline config for this model's outputs
                # Example: pass through threshold if an AI model downstream needs it
                if len(item.output_names) > 1 and len(item.input_names) > 1:
                     # This is a generic example, specific passthrough logic depends on pipeline design
                    passthrough_output_name = item.output_names[1]
                    passthrough_input_name = item.input_names[1] # Assuming a corresponding input
                    if passthrough_input_name in itemFuture:
                         await itemFuture.set_data(passthrough_output_name, itemFuture[passthrough_input_name])


            except FileNotFoundError as fnf_error:
                input_val = itemFuture.get(item.input_names[0], 'unknown_file_or_data')
                self.logger.error(f"File not found error processing image input '{str(input_val)}': {fnf_error}")
                if not itemFuture.done(): itemFuture.set_exception(fnf_error)
            except TypeError as type_error:
                input_val = itemFuture.get(item.input_names[0], 'unknown_file_or_data')
                self.logger.error(f"Type error processing image input '{str(input_val)}': {type_error}")
                if not itemFuture.done(): itemFuture.set_exception(type_error)
            except Exception as e:
                input_val = itemFuture.get(item.input_names[0], 'unknown_file_or_data')
                self.logger.error(f"An unexpected error occurred processing image input '{str(input_val)}': {e}", exc_info=True)
                if not itemFuture.done(): itemFuture.set_exception(e)

    async def load(self) -> None:
        # ImagePreprocessorModel typically doesn't load a "model" file.
        self.logger.info(f"ImagePreprocessorModel ({self.__class__.__name__}) is ready.")
        return
