import logging
from typing import Dict, Any, List, Optional

from ..async_processing.async_processor import ModelProcessor
from .ai_model import AIModel
from .torchscript_model import TorchScriptModel
from .python_callable_model import PythonCallableModel # Added
from .video_preprocessor import VideoPreprocessorModel
from .image_preprocessor import ImagePreprocessorModel
from .vlm_ai_model import VLMAIModel
from .base_model import Model as BaseModel # For type hinting actual model instances
from ..utils.exceptions import ConfigurationException, ModelNotFoundException

class ModelManager:
    def __init__(self, model_configurations: Dict[str, Dict[str, Any]]):
        """
        Initializes the ModelManager with all model configurations.

        Args:
            model_configurations: A dictionary where keys are unique model instance names
                                  and values are their configuration dictionaries.
        """
        self.model_configs: Dict[str, Dict[str, Any]] = model_configurations
        self.model_processors: Dict[str, ModelProcessor] = {} # Stores ModelProcessor instances
        self.logger: logging.Logger = logging.getLogger("logger")
        
        # This list tracks AIModel instances for potential cross-model adjustments (e.g., batch sizing)
        # It should store ModelProcessor instances whose .model is an AIModel
        self._active_ai_model_processors: List[ModelProcessor] = [] 

    def get_or_create_model_processor(self, model_instance_name: str) -> ModelProcessor:
        if model_instance_name not in self.model_processors:
            if model_instance_name not in self.model_configs:
                raise ModelNotFoundException(model_instance_name, 
                    f"Configuration for model instance '{model_instance_name}' not found.")
            
            config = self.model_configs[model_instance_name]
            try:
                model_processor_instance = self._model_factory(model_instance_name, config)
                self.model_processors[model_instance_name] = model_processor_instance
            except Exception as e:
                self.logger.error(f"Failed to create model processor for '{model_instance_name}': {e}", exc_info=True)
                raise ModelException(f"Failed to create model processor for '{model_instance_name}': {e}") from e
        
        return self.model_processors[model_instance_name]
    
    def refresh_model_processor(self, model_instance_name: str) -> ModelProcessor:
        """Re-creates and replaces a model processor. Useful if config needs reloading (though less common in package)."""
        if model_instance_name not in self.model_configs:
            raise ModelNotFoundException(model_instance_name,
                f"Configuration for model instance '{model_instance_name}' not found for refresh.")
        
        self.logger.info(f"Refreshing model processor for '{model_instance_name}'...")
        config = self.model_configs[model_instance_name]
        try:
            # If the model was an AI model, remove its old processor from tracking list
            if model_instance_name in self.model_processors:
                old_processor = self.model_processors[model_instance_name]
                if old_processor in self._active_ai_model_processors:
                    self._active_ai_model_processors.remove(old_processor)

            new_model_processor = self._model_factory(model_instance_name, config)
            self.model_processors[model_instance_name] = new_model_processor
            self._update_ai_model_batch_sizes() # Re-evaluate batch sizes
            self.logger.info(f"Successfully refreshed model processor for '{model_instance_name}'.")
            return new_model_processor
        except Exception as e:
            self.logger.error(f"Failed to refresh model processor for '{model_instance_name}': {e}", exc_info=True)
            raise ModelException(f"Failed to refresh model processor for '{model_instance_name}': {e}") from e

    def _model_factory(self, instance_name: str, model_config: Dict[str, Any]) -> ModelProcessor:
        model_type: Optional[str] = model_config.get("type")
        if not model_type:
            raise ConfigurationException(f"Model type not specified for model instance '{instance_name}'. Config: {model_config}")

        actual_model_instance: BaseModel # Type hint for the actual model (AIModel, VideoPreprocessorModel, etc.)
        
        # Add instance_name to config if not present, models might use it for logging/ID
        if 'instance_name' not in model_config:
            model_config['instance_name'] = instance_name

        match model_type:
            case "video_preprocessor":
                actual_model_instance = VideoPreprocessorModel(model_config)
            case "image_preprocessor":
                actual_model_instance = ImagePreprocessorModel(model_config)
            case "model": # Standard local AI model
                actual_model_instance = AIModel(model_config)
            case "vlm_model": # VLM client model
                actual_model_instance = VLMAIModel(model_config)
            case "torchscript": # For TorchScript models
                actual_model_instance = TorchScriptModel(
                    model_path=model_config["model_file_name"], # Expects model_file_name to be the path
                    batch_size=model_config.get("max_model_batch_size", model_config.get("max_batch_size", 1)), # Use more specific if available
                    device_str=model_config.get("device"),
                    fill_to_batch_size=model_config.get("fill_to_batch_size", True),
                    instance_name=instance_name
                )
            case "python_function": # For user-defined Python callables
                actual_model_instance = PythonCallableModel(model_config)
            case "result_coalescer":
                model_config["module_path"] = "haven_vlm_engine.models.common_model_functions"
                model_config["function_name"] = "result_coalescer"
                actual_model_instance = PythonCallableModel(model_config)
            case "result_finisher":
                model_config["module_path"] = "haven_vlm_engine.models.common_model_functions"
                model_config["function_name"] = "result_finisher"
                actual_model_instance = PythonCallableModel(model_config)
            case "batch_awaiter":
                model_config["module_path"] = "haven_vlm_engine.models.common_model_functions"
                model_config["function_name"] = "batch_awaiter"
                actual_model_instance = PythonCallableModel(model_config)
            case _:
                raise ConfigurationException(f"Model type '{model_type}' for instance '{instance_name}' not recognized.")
        
        model_processor = ModelProcessor(actual_model_instance)

        # Track AIModels for batch size updates
        if isinstance(actual_model_instance, AIModel): # This includes VLMAIModel if it inherits AIModel
            if model_processor not in self._active_ai_model_processors:
                 self._active_ai_model_processors.append(model_processor)
            self._update_ai_model_batch_sizes() # Update all after adding a new one

        return model_processor

    def _update_ai_model_batch_sizes(self) -> None:
        """
        Updates batch size related parameters for all active AIModel instances.
        This is called when a new AIModel is added or refreshed.
        """
        num_active_ai_models = len(self._active_ai_model_processors)
        if num_active_ai_models > 0:
            self.logger.debug(f"Updating batch sizes for {num_active_ai_models} active AI models.")
            for processor in self._active_ai_model_processors:
                # The model instance (AIModel) is at processor.model
                ai_model_instance = processor.model
                if hasattr(ai_model_instance, 'update_batch_with_mutli_models') and \
                   callable(ai_model_instance.update_batch_with_mutli_models):
                    ai_model_instance.update_batch_with_mutli_models(num_active_ai_models)
                    # ModelProcessor also needs to update its internal batch/queue sizes from the model
                    processor.update_values_from_child_model()
    
    def get_all_model_processors(self) -> List[ModelProcessor]:
        return list(self.model_processors.values())

    async def start_all_model_workers(self) -> None:
        """Starts worker tasks for all managed model processors."""
        self.logger.info("Starting workers for all managed models...")
        for name, processor in self.model_processors.items():
            try:
                self.logger.debug(f"Starting workers for model processor: {name}")
                await processor.start_workers()
            except Exception as e:
                self.logger.error(f"Failed to start workers for model processor '{name}': {e}", exc_info=True)
                # Decide if one failure should stop all, or just log and continue
                # For now, log and continue
        self.logger.info("Finished attempting to start all model workers.")
