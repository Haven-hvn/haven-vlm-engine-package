import logging
from typing import List, Dict, Any, Optional, Set, Union, Tuple

from ..async_processing.queue_item import QueueItem
from ..async_processing.item_future import ItemFuture
from ..models.ai_model import AIModel
from ..models.video_preprocessor import VideoPreprocessorModel
from ..models.vlm_ai_model import VLMAIModel
# These will be created/moved later:
from ..models.model_manager import ModelManager 
from .dynamic_ai_manager import DynamicAIManager
from .model_wrapper import ModelWrapper


logger: logging.Logger = logging.getLogger("logger")

def validate_string_list(input_list: Any) -> bool:
    if not isinstance(input_list, list):
        return False
    for item in input_list:
        if not isinstance(item, str):
            return False
    return True

class Pipeline:
    def __init__(self, 
                 pipeline_name: str, # Added pipeline_name for better context
                 configValues: Dict[str, Any], 
                 model_manager: ModelManager, 
                 dynamic_ai_manager: DynamicAIManager):
        
        self.pipeline_name = pipeline_name # Store the name

        if not validate_string_list(configValues.get("inputs")):
            raise ValueError(f"Pipeline '{pipeline_name}': Inputs must be a non-empty list of strings!")
        if not configValues.get("output") or not isinstance(configValues.get("output"), str):
            raise ValueError(f"Pipeline '{pipeline_name}': Output must be a non-empty string!")
        if not isinstance(configValues.get("models"), list) or not configValues.get("models"):
            raise ValueError(f"Pipeline '{pipeline_name}': Models must be a non-empty list!")
        
        self.short_name: Optional[str] = configValues.get("short_name")
        if self.short_name is None or not isinstance(self.short_name, str) or not self.short_name:
            raise ValueError(f"Pipeline '{pipeline_name}': short_name must be a non-empty string!")
        
        self.version: Optional[Union[float, str]] = configValues.get("version")
        if self.version is None or not (isinstance(self.version, (float, int)) or (isinstance(self.version, str) and self.version)):
            raise ValueError(f"Pipeline '{pipeline_name}': version must be a non-empty float or string representation of a number!")
        
        self.inputs: List[str] = configValues["inputs"]
        self.output: str = configValues["output"]
        self.models: List[ModelWrapper] = []

        model_config_entry: Dict[str, Any]
        for model_config_entry in configValues["models"]:
            if not validate_string_list(model_config_entry.get("inputs")):
                raise ValueError(f"Pipeline '{pipeline_name}', Model Config: Inputs must be a non-empty list of strings! Got: {model_config_entry.get('inputs')}")
            
            model_instance_name: Optional[str] = model_config_entry.get("name") # This is the instance name from pipeline config
            if not model_instance_name or not isinstance(model_instance_name, str):
                raise ValueError(f"Pipeline '{pipeline_name}', Model Config: 'name' (instance name) must be a non-empty string! Got: {model_instance_name}")
            
            model_inputs: List[str] = model_config_entry["inputs"]
            model_outputs: Union[str, List[str], None] = model_config_entry.get("outputs")

            # Validate model_outputs (can be string, list of strings, or None for some special models)
            is_valid_outputs = False
            if isinstance(model_outputs, str) and model_outputs:
                is_valid_outputs = True
            elif isinstance(model_outputs, list) and all(isinstance(o, str) and o for o in model_outputs):
                is_valid_outputs = True
            elif model_outputs is None: # Allowed for specific models like 'result_finisher' or dynamic groupers
                 # The original code had a pass for modelName != "result_finisher" if outputs were None/empty.
                 # This implies some models might not declare outputs if they are terminal or groupers.
                 # For now, we'll allow None, but pipeline execution needs to handle this.
                is_valid_outputs = True # Assuming None is valid for certain model types
            
            if not is_valid_outputs and model_instance_name not in ["result_finisher", "dynamic_video_ai", "dynamic_image_ai"]: # Allow None for specific types
                 # Stricter check: if model_outputs is None or empty list/str, it's an issue unless it's a special model type
                if model_outputs is None or (isinstance(model_outputs, (list,str)) and not model_outputs) :
                     logger.warning(f"Pipeline '{pipeline_name}', Model '{model_instance_name}': 'outputs' is None or empty. This is only allowed for specific model types.")
                     # For now, we'll proceed, but this could be an error depending on model type.
                     # The original code had a 'pass' here if modelName != "result_finisher".
                     # We'll assume None is acceptable for now and ModelWrapper can handle it.


            # Handle dynamic model groups first
            if model_instance_name == "dynamic_video_ai":
                # DynamicAIManager returns List[ModelWrapper]
                dynamic_model_wrappers: List[ModelWrapper] = dynamic_ai_manager.get_dynamic_video_ai_models(
                    model_inputs, 
                    model_outputs if isinstance(model_outputs, list) else [model_outputs] if isinstance(model_outputs, str) else []
                )
                self.models.extend(dynamic_model_wrappers)
                continue
            elif model_instance_name == "dynamic_image_ai":
                dynamic_model_wrappers: List[ModelWrapper] = dynamic_ai_manager.get_dynamic_image_ai_models(
                    model_inputs,
                    model_outputs if isinstance(model_outputs, list) else [model_outputs] if isinstance(model_outputs, str) else []
                )
                self.models.extend(dynamic_model_wrappers)
                continue
            
            # For regular models, get the ModelProcessor instance
            # model_instance_name here refers to the key in the main 'models' config dict provided to VLMEngine
            model_processor_instance: Any = model_manager.get_or_create_model_processor(model_instance_name)
            
            self.models.append(ModelWrapper(model_processor_instance, model_inputs, model_outputs, model_name_for_logging=model_instance_name))

        # Validate AI model categories (no overlaps)
        categories_set: Set[str] = set()
        for wrapper in self.models:
            # Access the actual model (e.g. AIModel) via model_processor.model
            actual_model = wrapper.model_processor.model 
            if isinstance(actual_model, AIModel): # Check if it's an AIModel instance
                current_categories = actual_model.model_category
                if isinstance(current_categories, str):
                    if current_categories in categories_set:
                        raise ValueError(f"Pipeline '{pipeline_name}': AI models must not have overlapping categories! Category: '{current_categories}'")
                    categories_set.add(current_categories)
                elif isinstance(current_categories, list):
                    for cat in current_categories:
                        if cat in categories_set:
                            raise ValueError(f"Pipeline '{pipeline_name}': AI models must not have overlapping categories! Category: '{cat}'")
                        categories_set.add(cat)
        
        # Configure VideoPreprocessorModels if VLM models are present
        is_vlm_pipeline: bool = any(isinstance(mw.model_processor.model, VLMAIModel) for mw in self.models)
        if is_vlm_pipeline:
            for model_wrapper in self.models:
                if isinstance(model_wrapper.model_processor.model, VideoPreprocessorModel):
                    model_wrapper.model_processor.model.set_vlm_pipeline_mode(True)
    
    async def event_handler(self, item_future: ItemFuture, key: str) -> None:
        if key == self.output: # Final output of the pipeline
            if key in item_future: # Check if the final output key is in the ItemFuture's data
                item_future.close_future(item_future[key])
            else:
                # This case (final output key not found) might indicate an issue or an alternative completion path.
                logger.warning(f"Pipeline '{self.pipeline_name}': Final output key '{key}' not found in ItemFuture data when expected.")
                # Optionally, set an error or a default value if the key is missing.
                # For now, if it's not there, the future won't be closed by this condition.
                # It might be closed by another mechanism or timeout.
                pass # Future remains open if key not found
        
        for current_model_wrapper in self.models:
            if key in current_model_wrapper.inputs: # If the newly set key is an input for a model
                # Check if all inputs for this model are now present in the item_future
                all_inputs_present = True
                for input_name in current_model_wrapper.inputs:
                    if input_name not in item_future: # Check presence in ItemFuture's data
                        all_inputs_present = False
                        break
                
                if all_inputs_present:
                    # All inputs are ready, add to this model's processing queue
                    await current_model_wrapper.model_processor.add_to_queue(
                        QueueItem(item_future, current_model_wrapper.inputs, current_model_wrapper.outputs)
                    )

    async def start_model_processing_workers(self) -> None:
        """Starts the worker tasks for all models in this pipeline."""
        for model_wrapper in self.models:
            # model_processor is the ModelProcessor instance
            if hasattr(model_wrapper.model_processor, 'start_workers') and callable(model_wrapper.model_processor.start_workers):
                 await model_wrapper.model_processor.start_workers()

    # Helper methods to get specific model types from the pipeline (if needed by external logic)
    def get_first_video_preprocessor(self) -> Optional[VideoPreprocessorModel]:
        for model_wrapper in self.models:
            if isinstance(model_wrapper.model_processor.model, VideoPreprocessorModel):
                return model_wrapper.model_processor.model
        return None
    
    def get_first_ai_model(self) -> Optional[AIModel]: # Could be AIModel or VLMAIModel
        for model_wrapper in self.models:
            if isinstance(model_wrapper.model_processor.model, AIModel): # Catches VLMAIModel too if it inherits AIModel
                return model_wrapper.model_processor.model
        return None
    
    def get_ai_models_info(self) -> List[Tuple[Optional[Union[str, float]], Optional[str], Optional[str], Optional[Union[str, List[str]]]]]:
        """Collects information from AIModel instances in the pipeline."""
        ai_info_list: List[Tuple[Optional[Union[str, float]], Optional[str], Optional[str], Optional[Union[str, List[str]]]]] = []
        for model_wrapper in self.models:
            actual_model = model_wrapper.model_processor.model
            if isinstance(actual_model, AIModel): # Includes VLMAIModel if it inherits
                version = getattr(actual_model, 'model_version', None)
                identifier = getattr(actual_model, 'model_identifier', None)
                file_name = getattr(actual_model, 'model_file_name', None) # May be an ID for VLMs
                category = getattr(actual_model, 'model_category', None)
                ai_info_list.append((version, identifier, file_name, category))
        return ai_info_list
