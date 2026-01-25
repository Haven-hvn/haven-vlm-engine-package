import logging
import time
import torch
from PIL import Image
import numpy as np
from importlib import import_module
import types
from .async_utils import ModelProcessor, QueueItem, ItemFuture
from .config_models import ModelConfig
from .preprocessing import preprocess_video
from .vlm_client import OpenAICompatibleVLMClient
from .multiplexer_vlm_client import MultiplexerVLMClient
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

# Placeholder for ModelRunner
ModelRunner = Any

class Model:
    def __init__(self, configValues: ModelConfig):
        self.max_queue_size: Optional[int] = configValues.max_queue_size
        self.max_batch_size: int = configValues.max_batch_size
        self.instance_count: int = configValues.instance_count
        self.logger: logging.Logger = logging.getLogger("logger")

    async def worker_function_wrapper(self, data: List[QueueItem]) -> None:
        try:
            await self.worker_function(data)
        except Exception as e:
            self.logger.error(f"Exception in worker_function: {e}", exc_info=True)
            item: QueueItem
            for item in data:
                if hasattr(item, 'item_future') and item.item_future:
                    item.item_future.set_exception(e)
                else:
                    self.logger.error("Item in batch lacks item_future, cannot propagate exception.")

    async def worker_function(self, data: List[QueueItem]) -> None:
        pass

    async def load(self) -> None:
        return

class AIModel(Model):
    def __init__(self, configValues: ModelConfig):
        super().__init__(configValues)
        self.model_file_name: Optional[str] = configValues.model_file_name
        self.model_license_name: Optional[str] = configValues.model_license_name
        self.model_threshold: Optional[float] = configValues.model_threshold
        self.model_return_tags: bool = configValues.model_return_tags
        self.model_return_confidence: bool = configValues.model_return_confidence
        self.device: Optional[str] = configValues.device
        self.fill_to_batch: bool = configValues.fill_to_batch_size
        self.model_image_size: Optional[Union[int, Tuple[int, int]]] = configValues.model_image_size
        self.model_category: Optional[Union[str, List[str]]] = configValues.model_category
        self.model_version: Optional[str] = configValues.model_version
        self.model_identifier: Optional[str] = configValues.model_identifier
        self.category_mappings: Optional[Dict[int, int]] = configValues.category_mappings
        self.normalization_config: Union[int, Dict[str, List[float]]] = configValues.normalization_config
        
        if self.model_file_name is None:
            raise ValueError("model_file_name is required for models of type model")
        if self.model_category is not None and isinstance(self.model_category, list) and len(self.model_category) > 1:
            if self.category_mappings is None:
                raise ValueError("category_mappings is required for models with more than one category")
        
        self.model: Optional[ModelRunner] = None
        self.tags: Dict[int, str] = {}

        if self.device is None:
            self.localdevice: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.localdevice: torch.device = torch.device(self.device)

    async def worker_function(self, data: List[QueueItem]) -> None:
        pass

    async def load(self) -> None:
        pass

class VLMAIModel(AIModel):
    def __init__(self, configValues: ModelConfig):
        super().__init__(configValues)
        self.client_config: ModelConfig = configValues
        self.vlm_model: Optional[Union[OpenAICompatibleVLMClient, MultiplexerVLMClient]] = None
        self.use_multiplexer: bool = configValues.use_multiplexer

    async def worker_function(self, data: List[QueueItem]):
        self.logger.info(f"VLMAIModel worker_function called with {len(data)} items")
        for item in data:
            itemFuture: ItemFuture = item.item_future
            try:
                image_tensor: Any = itemFuture[item.input_names[0]]
                threshold: float = itemFuture[item.input_names[1]] if item.input_names[1] in itemFuture else self.model_threshold
                return_confidence: bool = itemFuture[item.input_names[2]] if item.input_names[2] in itemFuture else self.model_return_confidence

                image_np: np.ndarray
                if hasattr(image_tensor, 'cpu') and hasattr(image_tensor, 'numpy'):
                    image_np = image_tensor.cpu().numpy()
                elif isinstance(image_tensor, np.ndarray):
                    image_np = image_tensor
                elif isinstance(image_tensor, Image.Image):
                    image_pil: Image.Image = image_tensor
                else:
                    raise TypeError(f"Unsupported image_tensor type: {type(image_tensor)}")

                if not isinstance(image_tensor, Image.Image):
                    if image_np.ndim == 3 and image_np.shape[0] == 3:
                        image_np = np.transpose(image_np, (1, 2, 0))
                    elif image_np.ndim == 2:
                        image_np = np.stack((image_np,)*3, axis=-1)

                    if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                        if image_np.min() >= 0 and image_np.max() <= 1:
                            image_np = (image_np * 255)
                    
                    image_np = image_np.astype(np.uint8)
                    image_pil: Image.Image = Image.fromarray(image_np)
                
                scores: Dict[str, float] = await self.vlm_model.analyze_frame(image_pil)
                self.logger.debug(f"VLM scores for frame: {scores}")
                
                tags = []
                for tag_name, confidence in scores.items():
                    if confidence > threshold:
                        if return_confidence:
                            tags.append((tag_name, round(confidence, 2)))
                        else:
                            tags.append(tag_name)
                
                self.logger.debug(f"Tags detected with threshold {threshold}: {tags}")
                
                if isinstance(self.model_category, list):
                    for category in self.model_category:
                        await itemFuture.set_data(category, tags)
                else:
                    await itemFuture.set_data(self.model_category, tags)
            except Exception as e:
                itemFuture.set_exception(e)

    async def load(self) -> None:
        if self.vlm_model is None:
            self.logger.info(f"Loading VLMAIModel with config: {self.client_config.dict().keys()}")

            if self.use_multiplexer:
                self.logger.info("Using MultiplexerVLMClient for load balancing across multiple endpoints")
                self.vlm_model = MultiplexerVLMClient(config=self.client_config.dict())
                # Initialize the multiplexer
                await self.vlm_model._ensure_initialized()
            else:
                self.logger.info("Using single endpoint OpenAICompatibleVLMClient")
                self.vlm_model = OpenAICompatibleVLMClient(config=self.client_config.dict())

            self.logger.info("VLMAIModel loaded successfully")

class PythonModel(Model):
    def __init__(self, configValues: ModelConfig):
        super().__init__(configValues)
        self.function_name: Optional[str] = configValues.model_file_name
        if self.function_name is None:
            raise ValueError("function_name is required for models of type python")
        module_name: str = "vlm_engine.python_functions"
        try:
            module: types.ModuleType = import_module(module_name)
            self.function: Callable[[List[QueueItem]], None] = getattr(module, self.function_name)
        except ImportError:
            raise ImportError(f"Module '{module_name}' not found.")
        except AttributeError:
            raise AttributeError(f"Function '{self.function_name}' not found in module '{module_name}'.")

    async def worker_function(self, data: List[QueueItem]) -> None:
        await self.function(data)

class VideoPreprocessorModel(Model):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.logger = logging.getLogger("logger")
        self.device: str = model_config.device or "cpu"
        self.image_size: Union[int, List[int], Tuple[int, int]] = model_config.model_image_size or 512
        self.frame_interval: float = 0.5 # Default value
        self.use_half_precision: bool = True # Default value
        self.normalization_config: Union[int, Dict[str, List[float]]] = model_config.normalization_config
        self.process_for_vlm: bool = False
    
    def set_vlm_pipeline_mode(self, mode: bool) -> None:
        self.process_for_vlm = mode
        self.logger.info(f"VideoPreprocessorModel VLM mode set to: {self.process_for_vlm}")

    async def worker_function(self, queue_items: List[QueueItem]) -> None:
        for item in queue_items:
            itemFuture: ItemFuture = item.item_future
            try:
                video_path: str = itemFuture[item.input_names[0]]
                use_timestamps: bool = itemFuture[item.input_names[1]]
                frame_interval_override: Optional[float] = itemFuture[item.input_names[2]]
                current_frame_interval: float = frame_interval_override if frame_interval_override is not None else self.frame_interval
                vr_video: bool = itemFuture[item.input_names[5]]
                
                children: List[ItemFuture] = []
                processed_frames_count: int = 0
                
                for frame_index, frame_tensor in preprocess_video(video_path, current_frame_interval, self.image_size, self.use_half_precision, self.device, use_timestamps, vr_video=vr_video, norm_config_idx=self.normalization_config, process_for_vlm=self.process_for_vlm):
                    processed_frames_count += 1
                    
                    future_data_payload: Dict[str, Any] = {
                        "dynamic_frame": frame_tensor, 
                        "frame_index": frame_index,
                        "dynamic_threshold": itemFuture[item.input_names[3]],
                        "dynamic_return_confidence": itemFuture[item.input_names[4]],
                        "dynamic_skipped_categories": itemFuture[item.input_names[6]]
                    }
                    result_future: ItemFuture = await ItemFuture.create(item, future_data_payload, item.item_future.handler)
                    
                    # Set frame_index in the result future so result_coalescer can find it
                    await result_future.set_data("frame_index", frame_index)
                    
                    children.append(result_future)
                
                await itemFuture.set_data(item.output_names[0], children)
            except Exception as e:
                itemFuture.set_exception(e)

class ModelManager:
    def __init__(self, models_config: Dict[str, ModelConfig]):
        self.models_config = models_config
        self.models: Dict[str, ModelProcessor] = {}
        self.logger: logging.Logger = logging.getLogger("logger")
        self.ai_models: List[ModelProcessor] = []

    def get_or_create_model(self, modelName: str) -> ModelProcessor:
        if modelName not in self.models:
            created_model: Optional[ModelProcessor] = self.create_model(modelName)
            if created_model is None:
                raise ValueError(f"Failed to create model: {modelName}")
            self.models[modelName] = created_model
        return self.models[modelName]
    
    def create_model(self, modelName: str) -> Optional[ModelProcessor]:
        if modelName not in self.models_config:
            raise ValueError(f"Model '{modelName}' not found in configuration.")
        
        model_config = self.models_config[modelName]
        model_processor_instance: ModelProcessor = self.model_factory(model_config)
        return model_processor_instance
    
    def model_factory(self, model_config: ModelConfig) -> ModelProcessor:
        model_type: str = model_config.type
        
        model_instance: Any
        match model_type:
            case "video_preprocessor":
                model_instance = VideoPreprocessorModel(model_config)
                return ModelProcessor(model_instance)
            case "binary_search_processor":
                # New binary search processor for optimized video processing
                from .binary_search_processor import BinarySearchProcessor
                model_instance = BinarySearchProcessor(model_config)
                return ModelProcessor(model_instance)
            case "vlm_model":
                model_instance = VLMAIModel(model_config)
                model_processor: ModelProcessor = ModelProcessor(model_instance)
                self.ai_models.append(model_processor)
                return model_processor
            case "python":
                model_instance = PythonModel(model_config)
                return ModelProcessor(model_instance)
            case _:
                raise ValueError(f"Model type '{model_type}' not recognized!")
