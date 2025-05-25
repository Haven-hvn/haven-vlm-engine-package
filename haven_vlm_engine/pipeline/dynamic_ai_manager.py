import logging
from typing import List, Dict, Any, Optional, Union, Tuple

from ..models.ai_model import AIModel # Base class for type checking
from ..models.base_model import Model as BaseModel # Actual model instance type
from ..models.model_manager import ModelManager
from .model_wrapper import ModelWrapper
from ..utils.exceptions import NoActiveModelsException # To be created
from ..async_processing.async_processor import ModelProcessor


class DynamicAIManager:
    def __init__(self, 
                 model_manager: ModelManager, 
                 active_ai_model_names: List[str] # Names of models to be used dynamically
                ):
        self.model_manager: ModelManager = model_manager
        self.active_ai_model_names: List[str] = active_ai_model_names # Names of model instances configured in the main engine
        
        self.loaded: bool = False
        self.image_size: Optional[Union[Tuple[int, int], List[int]]] = None # type: ignore
        self.normalization_config: Optional[Union[int, Dict[str, List[float]]]] = None # type: ignore
        self.active_model_processors: List[ModelProcessor] = [] # Stores ModelProcessor instances
        self.logger: logging.Logger = logging.getLogger("logger")

    def load_and_verify_models(self) -> None:
        if self.loaded:
            return
        
        if not self.active_ai_model_names:
            self.logger.error("Error: No active AI model names provided for DynamicAIManager.")
            raise NoActiveModelsException("No active AI model names provided for DynamicAIManager.")

        temp_model_processors: List[ModelProcessor] = []
        for model_instance_name in self.active_ai_model_names:
            # get_or_create_model_processor returns a ModelProcessor instance
            model_processor = self.model_manager.get_or_create_model_processor(model_instance_name)
            temp_model_processors.append(model_processor)
        
        self._verify_model_compatibility(temp_model_processors) # Pass ModelProcessor list
        self.active_model_processors = temp_model_processors
        self.loaded = True
        self.logger.info(f"DynamicAIManager loaded and verified {len(self.active_model_processors)} active AI models.")

    def _get_model_processor_by_name(self, name: str) -> ModelProcessor:
        """Helper to get a specific model processor by its configured instance name."""
        return self.model_manager.get_or_create_model_processor(name)

    def get_dynamic_video_ai_models(self, inputs: List[str], outputs: List[str]) -> List[ModelWrapper]:
        if not self.loaded:
            self.load_and_verify_models()
        
        model_wrappers: List[ModelWrapper] = []

        # 1. Video Preprocessor
        # Assumes a model named "video_preprocessor_dynamic" is configured in the main model_manager
        video_preprocessor_processor = self._get_model_processor_by_name("video_preprocessor_dynamic")
        # Configure the actual VideoPreprocessorModel instance via its processor
        if hasattr(video_preprocessor_processor.model, 'image_size'):
            video_preprocessor_processor.model.image_size = self.image_size # type: ignore
        if hasattr(video_preprocessor_processor.model, 'normalization_config'):
             video_preprocessor_processor.model.normalization_config = self.normalization_config # type: ignore

        model_wrappers.append(ModelWrapper(video_preprocessor_processor, inputs, 
                                         ["dynamic_children", "dynamic_frame", "frame_index", "dynamic_threshold", "dynamic_return_confidence", "dynamic_skipped_categories"], 
                                         model_name_for_logging="video_preprocessor_dynamic"))

        # 2. Active AI Models (from self.active_model_processors)
        for model_processor_instance in self.active_model_processors:
            actual_model = model_processor_instance.model # This is the AIModel instance
            log_name = getattr(actual_model, 'model_identifier', None) or \
                       getattr(actual_model, 'model_file_name', f"active_ai_model_{self.active_model_processors.index(model_processor_instance)}")
            
            # Ensure model_category is a list for extend, even if it's a single string
            model_categories = actual_model.model_category
            output_categories = [model_categories] if isinstance(model_categories, str) else model_categories
            if not output_categories: # Should not happen if validated
                output_categories = [f"output_for_{log_name}"]


            model_wrappers.append(ModelWrapper(model_processor_instance, 
                                             ["dynamic_frame", "dynamic_threshold", "dynamic_return_confidence", "dynamic_skipped_categories"], 
                                             output_categories, 
                                             model_name_for_logging=str(log_name)))

        # 3. Result Coalescer
        coalescer_processor = self._get_model_processor_by_name("result_coalescer")
        coalesce_inputs: List[str] = ["frame_index"] # frame_index is always first
        for mp_instance in self.active_model_processors:
            categories = mp_instance.model.model_category
            if isinstance(categories, list):
                coalesce_inputs.extend(categories)
            elif isinstance(categories, str):
                coalesce_inputs.append(categories)
        
        model_wrappers.append(ModelWrapper(coalescer_processor, 
                                         coalesce_inputs, 
                                         ["dynamic_coalesced_result"], # Coalescer has a defined output
                                         model_name_for_logging="result_coalescer"))

        # 4. Result Finisher (optional, might not have explicit outputs in pipeline def)
        finisher_processor = self._get_model_processor_by_name("result_finisher")
        model_wrappers.append(ModelWrapper(finisher_processor, 
                                         ["dynamic_coalesced_result"], 
                                         [], # Finisher might not produce a named output for further steps
                                         model_name_for_logging="result_finisher"))

        # 5. Batch Awaiter (handles the list of child futures from preprocessor)
        awaiter_processor = self._get_model_processor_by_name("batch_awaiter")
        model_wrappers.append(ModelWrapper(awaiter_processor, 
                                         ["dynamic_children"], # Input is the list of futures
                                         outputs, # Final outputs of the dynamic group
                                         model_name_for_logging="batch_awaiter"))
        
        self.logger.debug("Finished creating dynamic Video AI model wrappers.")
        return model_wrappers
    
    def get_dynamic_image_ai_models(self, inputs: List[str], outputs: List[str]) -> List[ModelWrapper]:
        if not self.loaded:
            self.load_and_verify_models()

        model_wrappers: List[ModelWrapper] = []

        # 1. Image Preprocessor
        image_preprocessor_processor = self._get_model_processor_by_name("image_preprocessor_dynamic")
        if hasattr(image_preprocessor_processor.model, 'image_size'):
            image_preprocessor_processor.model.image_size = self.image_size # type: ignore
        if hasattr(image_preprocessor_processor.model, 'normalization_config'):
            image_preprocessor_processor.model.normalization_config = self.normalization_config # type: ignore
        
        # Input for image preprocessor is typically the first item from the main pipeline inputs (e.g., image_path)
        model_wrappers.append(ModelWrapper(image_preprocessor_processor, 
                                         [inputs[0]], 
                                         ["dynamic_processed_image"], # Output of preprocessor
                                         model_name_for_logging="image_preprocessor_dynamic"))

        # 2. Active AI Models
        ai_model_outputs_collector: List[str] = []
        for model_processor_instance in self.active_model_processors:
            actual_model = model_processor_instance.model
            log_name = getattr(actual_model, 'model_identifier', None) or \
                       getattr(actual_model, 'model_file_name', f"active_ai_model_{self.active_model_processors.index(model_processor_instance)}")
            
            # Inputs for AI models: processed image + other inputs like threshold, return_confidence
            # These other inputs (inputs[1], inputs[2], etc.) are passed from the main pipeline call.
            ai_model_inputs = ["dynamic_processed_image"] + inputs[1:] # Combine processed image with other pipeline inputs

            model_categories = actual_model.model_category
            current_model_output_names = [model_categories] if isinstance(model_categories, str) else model_categories
            if not current_model_output_names: current_model_output_names = [f"output_for_{log_name}"]
            
            ai_model_outputs_collector.extend(current_model_output_names)

            model_wrappers.append(ModelWrapper(model_processor_instance, 
                                             ai_model_inputs, 
                                             current_model_output_names, 
                                             model_name_for_logging=str(log_name)))

        # 3. Result Coalescer (if multiple AI models or complex outputs)
        # The final output of this dynamic group is specified by 'outputs' param
        coalescer_processor = self._get_model_processor_by_name("result_coalescer")
        model_wrappers.append(ModelWrapper(coalescer_processor, 
                                         ai_model_outputs_collector, # All outputs from AI models
                                         outputs, # Final outputs of the dynamic group
                                         model_name_for_logging="result_coalescer"))

        self.logger.debug("Finished creating dynamic Image AI model wrappers.")
        return model_wrappers
    
    def _verify_model_compatibility(self, model_processors: List[ModelProcessor]) -> None:
        """ Verifies that all dynamically loaded AI models have compatible image sizes and normalization. """
        current_image_size: Optional[Union[Tuple[int, int], List[int]]] = None # type: ignore
        current_norm_config: Optional[Union[int, Dict[str, List[float]]]] = None # type: ignore
        
        for processor_instance in model_processors:
            # The actual model (e.g., AIModel instance) is at processor_instance.model
            actual_model = processor_instance.model
            if not isinstance(actual_model, AIModel): # Ensure it's an AIModel or its subclass
                raise ValueError(f"Dynamic AI group error: Model '{getattr(actual_model, 'model_identifier', 'Unknown')}' is not an AIModel derivative.")

            # Check for required attributes directly on the AIModel instance
            if not hasattr(actual_model, 'model_category') or actual_model.model_category is None:
                raise ValueError(f"Dynamic AI model '{actual_model.model_identifier}' must have 'model_category' set.")
            if not hasattr(actual_model, 'model_version') or actual_model.model_version is None:
                raise ValueError(f"Dynamic AI model '{actual_model.model_identifier}' must have 'model_version' set.")
            if not hasattr(actual_model, 'model_image_size') or actual_model.model_image_size is None:
                raise ValueError(f"Dynamic AI model '{actual_model.model_identifier}' must have 'model_image_size' set.")
            if not hasattr(actual_model, 'normalization_config') or actual_model.normalization_config is None:
                 raise ValueError(f"Dynamic AI model '{actual_model.model_identifier}' must have 'normalization_config' set.")


            if current_image_size is None:
                current_image_size = actual_model.model_image_size
            elif current_image_size != actual_model.model_image_size:
                raise ValueError(f"Dynamic AI models must all have the same 'model_image_size'. "
                                 f"Model '{actual_model.model_identifier}' (size: {actual_model.model_image_size}) "
                                 f"differs from established size: {current_image_size}.")
            
            if current_norm_config is None:
                current_norm_config = actual_model.normalization_config
            elif current_norm_config != actual_model.normalization_config:
                raise ValueError(f"Dynamic AI models must all have the same 'normalization_config'. "
                                 f"Model '{actual_model.model_identifier}' (config: {actual_model.normalization_config}) "
                                 f"differs from established config: {current_norm_config}.")
        
        self.image_size = current_image_size
        self.normalization_config = current_norm_config
        self.logger.debug("Successfully verified compatibility of dynamic AI models.")
