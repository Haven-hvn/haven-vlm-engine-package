import logging
import time
import torch
from PIL import Image
import numpy as np
from importlib import import_module
import types
from .config_models import ModelConfig
from .preprocessing import preprocess_video
from .vlm_client import OpenAICompatibleVLMClient
from .multiplexer_vlm_client import MultiplexerVLMClient
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .async_utils import ModelProcessor, QueueItem, ItemFuture
    from .models import VLMAIModel

class Model:
    def __init__(self, configValues: ModelConfig):
        self.max_queue_size: Optional[int] = configValues.max_queue_size
        self.max_batch_size: int = configValues.max_batch_size
        self.instance_count: int = configValues.instance_count
        self.logger: logging.Logger = logging.getLogger("logger")

    async def worker_function_wrapper(self, data: List['QueueItem']) -> None:
        try:
            await self.worker_function(data)
        except Exception as e:
            self.logger.error(f"Exception in worker_function: {e}", exc_info=True)
            item: 'QueueItem'
            for item in data:
                if hasattr(item, 'item_future') and item.item_future:
                    item.item_future.set_exception(e)
                else:
                    self.logger.error("Item in batch lacks item_future, cannot propagate exception.")

    async def worker_function(self, data: List['QueueItem']) -> None:
        pass

    async def load(self) -> None:
        return

class VLMAIModel(Model):
    def __init__(self, configValues: ModelConfig):
        super().__init__(configValues)
        self.model_return_tags: bool = configValues.model_return_tags
        self.model_return_confidence: bool = configValues.model_return_confidence
        self.fill_to_batch: bool = configValues.fill_to_batch_size
        self.model_image_size: Optional[Union[int, Tuple[int, int]]] = configValues.model_image_size
        self.model_category: Optional[Union[str, List[str]]] = configValues.model_category
        self.category_mappings: Optional[Dict[int, int]] = configValues.category_mappings
        self.normalization_config: Union[int, Dict[str, List[float]]] = configValues.normalization_config
        
        self.client_config: ModelConfig = configValues
        self.vlm_model: Optional[Union[OpenAICompatibleVLMClient, MultiplexerVLMClient]] = None
        self.use_multiplexer: bool = configValues.use_multiplexer
        self.tags: Dict[int, str] = {}

    async def worker_function(self, data: List['QueueItem']):
        self.logger.info(f"VLMAIModel worker_function called with {len(data)} items")
        for item in data:
            itemFuture: ItemFuture = item.item_future
            try:
                self.logger.debug(f"[DEBUG_VLM] VLMAIModel processing item, input_names: {item.input_names}, output_names: {item.output_names}")
                self.logger.debug(f"[DEBUG_VLM] ItemFuture data keys available: {list(itemFuture.data.keys()) if itemFuture.data else 'None'}")
                self.logger.debug(f"[DEBUG_VLM] Looking for input_names[0]='{item.input_names[0]}' in ItemFuture")

                input_name = item.input_names[0]
                image_tensor: Any = itemFuture[input_name]
                self.logger.debug(f"[DEBUG_VLM] Retrieved image_tensor type: {type(image_tensor)}, value: {image_tensor if not isinstance(image_tensor, torch.Tensor) else '<Tensor>'}")

                threshold: float = itemFuture[item.input_names[1]] if item.input_names[1] in itemFuture else 0.5
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
        self.logger.debug(f"[DEBUG_PYTHON_MODEL] PythonModel.__init__ called, configValues type: {type(configValues).__name__}")
        self.function_name: Optional[str] = configValues.function_name
        self.logger.debug(f"[DEBUG_PYTHON_MODEL] function_name from configValues: {repr(self.function_name)}")
        if self.function_name is None:
            self.logger.error(f"[DEBUG_PYTHON_MODEL] ERROR: function_name is None! Full config: {configValues.dict()}")
            raise ValueError("function_name is required for models of type python")
        if self.function_name == "":
            self.logger.error(f"[DEBUG_PYTHON_MODEL] ERROR: function_name is empty string! Full config: {configValues.dict()}")
        module_name: str = "vlm_engine.python_functions"
        try:
            self.logger.debug(f"[DEBUG_PYTHON_MODEL] Importing module '{module_name}' for function '{self.function_name}'")
            module: types.ModuleType = import_module(module_name)
            self.function: Callable[[List['QueueItem']], None] = getattr(module, self.function_name)
            self.logger.debug(f"[DEBUG_PYTHON_MODEL] Successfully imported function '{self.function_name}' from module '{module_name}'")
        except ImportError:
            self.logger.error(f"[DEBUG_PYTHON_MODEL] ImportError: Module '{module_name}' not found.")
            raise ImportError(f"Module '{module_name}' not found.")
        except AttributeError:
            self.logger.error(f"[DEBUG_PYTHON_MODEL] AttributeError: Function '{self.function_name}' not found in module '{module_name}'.")
            raise AttributeError(f"Function '{self.function_name}' not found in module '{module_name}'.")

    async def worker_function(self, data: List['QueueItem']) -> None:
        await self.function(data)

class VideoPreprocessorModel(Model):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.logger = logging.getLogger("logger")
        self.image_size: Union[int, List[int], Tuple[int, int]] = model_config.model_image_size or 512
        self.frame_interval: float = 0.5 # Default value
        self.use_half_precision: bool = True # Default value
        self.device: str = model_config.device or "cpu"
        self.normalization_config: Union[int, Dict[str, List[float]]] = model_config.normalization_config
        self.process_for_vlm: bool = False
    
    def set_vlm_pipeline_mode(self, mode: bool) -> None:
        self.process_for_vlm = mode
        self.logger.info(f"VideoPreprocessorModel VLM mode set to: {self.process_for_vlm}")

    async def worker_function(self, queue_items: List['QueueItem']) -> None:
        for item in queue_items:
            itemFuture: ItemFuture = item.item_future
            try:
                self.logger.debug(f"VideoPreprocessorModel processing item, input_names: {item.input_names}")
                video_path: str = itemFuture[item.input_names[0]]
                self.logger.debug(f"Video path: {video_path}, type: {type(video_path)}")
                use_timestamps: bool = itemFuture[item.input_names[1]]
                frame_interval_override: Optional[float] = itemFuture[item.input_names[2]]
                current_frame_interval: float = frame_interval_override if frame_interval_override is not None else self.frame_interval
                vr_video: bool = itemFuture[item.input_names[5]]
                
                children: List[ItemFuture] = []
                processed_frames_count: int = 0
                
                for frame_index, frame_tensor in preprocess_video(video_path, current_frame_interval, self.image_size, self.use_half_precision, self.device, use_timestamps, vr_video=vr_video, norm_config_idx=self.normalization_config, process_for_vlm=self.process_for_vlm):
                    processed_frames_count += 1
                    
                    self.logger.debug(f"Creating child for frame {frame_index}, frame_tensor type: {type(frame_tensor)}")
                    
                    future_data_payload: Dict[str, Any] = {
                        "dynamic_frame": frame_tensor, 
                        "frame_index": frame_index,
                        "dynamic_threshold": itemFuture[item.input_names[3]],
                        "dynamic_return_confidence": itemFuture[item.input_names[4]],
                        "dynamic_skipped_categories": itemFuture[item.input_names[6]]
                    }
                    result_future: ItemFuture = await ItemFuture.create(itemFuture, future_data_payload, itemFuture.handler)
                    
                    self.logger.debug(f"Created child ItemFuture with data keys: {list(future_data_payload.keys())}")
                    
                    # Set frame_index in the result future so result_coalescer can find it
                    await result_future.set_data("frame_index", frame_index)
                    
                    children.append(result_future)
                
                await itemFuture.set_data(item.output_names[0], children)
            except Exception as e:
                itemFuture.set_exception(e)

class ModelManager:
    def __init__(self, models_config: Dict[str, ModelConfig]):
        self.models_config = models_config
        self.models: Dict[str, 'ModelProcessor'] = {}
        self.logger: logging.Logger = logging.getLogger("logger")
        self.ai_models: List['ModelProcessor'] = []

    def get_or_create_model(self, modelName: str) -> 'ModelProcessor':
        if modelName not in self.models:
            created_model: Optional[ModelProcessor] = self.create_model(modelName)
            if created_model is None:
                raise ValueError(f"Failed to create model: {modelName}")
            self.models[modelName] = created_model
        return self.models[modelName]
    
    def create_model(self, modelName: str) -> Optional['ModelProcessor']:
        self.logger.debug(f"[DEBUG_CREATE_MODEL] Creating model: {modelName}")
        if modelName not in self.models_config:
            self.logger.error(f"[DEBUG_CREATE_MODEL] Model '{modelName}' not found in configuration!")
            raise ValueError(f"Model '{modelName}' not found in configuration.")
        
        model_config = self.models_config[modelName]
        self.logger.debug(f"[DEBUG_CREATE_MODEL] Model config for {modelName}: type={model_config.type}, function_name={repr(getattr(model_config, 'function_name', 'NO_ATTR'))}")
        
        model_processor_instance: 'ModelProcessor' = self.model_factory(model_config)
        self.logger.debug(f"[DEBUG_CREATE_MODEL] Successfully created model: {modelName}")
        return model_processor_instance
    
    def model_factory(self, model_config: ModelConfig) -> 'ModelProcessor':
        # Import here to avoid circular dependency
        from .async_utils import ModelProcessor
        
        model_type: str = model_config.type
        self.logger.debug(f"[DEBUG_MODEL_FACTORY] Creating model of type: {model_type}")
        self.logger.debug(f"[DEBUG_MODEL_FACTORY] Full model config: {model_config.dict()}")
        
        if model_type == 'python':
            self.logger.debug(f"[DEBUG_MODEL_FACTORY] Python model detected! function_name: {repr(model_config.function_name)}")
            if model_config.function_name is None:
                self.logger.error(f"[DEBUG_MODEL_FACTORY] ⚠️ CRITICAL: Python model with None function_name!")
            elif model_config.function_name == "":
                self.logger.error(f"[DEBUG_MODEL_FACTORY] ⚠️ CRITICAL: Python model with empty string function_name!")
        
        model_instance: Any
        match model_type:
            case "video_preprocessor":
                self.logger.debug(f"[DEBUG_MODEL_FACTORY] Creating VideoPreprocessorModel")
                model_instance = VideoPreprocessorModel(model_config)
                return ModelProcessor(model_instance)
            case "binary_search_processor":
                # New binary search processor for optimized video processing
                self.logger.debug(f"[DEBUG_MODEL_FACTORY] Creating BinarySearchProcessor")
                from .binary_search_processor import BinarySearchProcessor
                model_instance = BinarySearchProcessor(model_config)
                return ModelProcessor(model_instance)
            case "vlm_model":
                self.logger.debug(f"[DEBUG_MODEL_FACTORY] Creating VLMAIModel")
                model_instance = VLMAIModel(model_config)
                model_processor: ModelProcessor = ModelProcessor(model_instance)
                self.ai_models.append(model_processor)
                return model_processor
            case "python":
                self.logger.debug(f"[DEBUG_MODEL_FACTORY] Creating PythonModel with function_name='{model_config.function_name}'")
                model_instance = PythonModel(model_config)
                return ModelProcessor(model_instance)
            case _:
                self.logger.error(f"[DEBUG_MODEL_FACTORY] Unrecognized model type: {model_type}")
                raise ValueError(f"Model type '{model_type}' not recognized!")
