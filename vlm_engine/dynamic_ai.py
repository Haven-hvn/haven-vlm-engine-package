import logging
from .models import ModelManager, AIModel
from .model_wrapper import ModelWrapper
from typing import List, Dict, Any, Optional, Union, Tuple

class DynamicAIManager:
    def __init__(self, model_manager: ModelManager, active_ai_models: List[str]):
        self.model_manager: ModelManager = model_manager
        self.ai_model_names: List[str] = active_ai_models
        self.loaded: bool = False
        self.image_size: Optional[Union[Tuple[int, int], List[int]]] = None
        self.normalization_config: Optional[Dict[str, Any]] = None
        self.models: List[AIModel] = []
        self.logger: logging.Logger = logging.getLogger("logger")

    def load(self) -> None:
        if self.loaded:
            return
        models: List[AIModel] = []

        if self.ai_model_names is None or len(self.ai_model_names) == 0:
            raise Exception("Error: No active AI models found in configuration.")
        for model_name in self.ai_model_names:
            # We only want to load the AI models, not all models
            model = self.model_manager.get_or_create_model(model_name)
            if isinstance(model.model, AIModel):
                models.append(model.model)
        self.__verify_models(models)
        self.models = models
        self.loaded = True

    def get_dynamic_video_ai_models(self, inputs: List[str], outputs: List[str]) -> List[ModelWrapper]:
        self.load()
        model_wrappers: List[ModelWrapper] = []
        
        # Use binary search processor for optimized video processing
        binary_search_processor = self.model_manager.get_or_create_model("binary_search_processor_dynamic")
        if hasattr(binary_search_processor.model, 'image_size'):
            binary_search_processor.model.image_size = self.image_size
        if hasattr(binary_search_processor.model, 'normalization_config'):
            binary_search_processor.model.normalization_config = self.normalization_config

        model_wrappers.append(ModelWrapper(binary_search_processor, inputs, 
                                         ["dynamic_children", "dynamic_frame", "frame_index", "dynamic_threshold", "dynamic_return_confidence", "dynamic_skipped_categories"], 
                                         model_name_for_logging="binary_search_processor_dynamic"))

        for model_idx, model in enumerate(self.models):
            # Need to get the ModelProcessor wrapper for the AIModel
            model_processor = self.model_manager.get_or_create_model(self.ai_model_names[model_idx])
            log_name = getattr(model, 'model_identifier', None) or \
                         getattr(model, 'model_file_name', f"active_ai_model_{model_idx}")
            model_wrappers.append(ModelWrapper(model_processor, 
                                             ["dynamic_frame", "dynamic_threshold", "dynamic_return_confidence", "dynamic_skipped_categories"], 
                                             model.model_category, 
                                             model_name_for_logging=str(log_name)))

        coalesce_inputs: List[str] = []
        for model in self.models:
            categories: Union[str, List[str]] = model.model_category
            if isinstance(categories, list):
                coalesce_inputs.extend(categories)
            else:
                coalesce_inputs.append(categories)
        coalesce_inputs.insert(0, "frame_index")
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("result_coalescer"), 
                                         coalesce_inputs, 
                                         ["dynamic_result"],
                                         model_name_for_logging="result_coalescer"))

        # Add result finisher for the coalesced result
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("result_finisher"), 
                                         ["dynamic_result"], 
                                         [], 
                                         model_name_for_logging="result_finisher"))

        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("batch_awaiter"), 
                                         ["dynamic_children"], 
                                         ["frame_results"], 
                                         model_name_for_logging="batch_awaiter"))
        
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("video_result_postprocessor"),
                                         ["frame_results", "video_path", "time_interval", "threshold", "existing_video_data"],
                                         outputs,
                                         model_name_for_logging="video_result_postprocessor"))
        return model_wrappers
    
    def __verify_models(self, models: List[AIModel]) -> None:
        current_image_size: Optional[Union[Tuple[int, int], List[int]]] = None
        current_norm_config: Optional[Dict[str, Any]] = None
        
        for model_instance in models:
            if not isinstance(model_instance, AIModel):
                raise ValueError(f"Error: Dynamic AI models must all be AI models! {model_instance} is not an AI model!")
            
            if current_image_size is None:
                current_image_size = model_instance.model_image_size
            elif current_image_size != model_instance.model_image_size:
                raise ValueError(f"Error: Dynamic AI models must all have the same model_image_size! {model_instance} has a different model_image_size than other models!")
            
            if current_norm_config is None:
                current_norm_config = model_instance.normalization_config
            elif current_norm_config != model_instance.normalization_config:
                raise ValueError(f"Error: Dynamic AI models must all have the same normalization_config! {model_instance} has a different normalization_config than other models!")
        self.image_size = current_image_size
        self.normalization_config = current_norm_config
