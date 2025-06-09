import logging
from .async_utils import ItemFuture, QueueItem
from .models import ModelManager, AIModel, VLMAIModel, VideoPreprocessorModel
from .config_models import PipelineConfig, PipelineModelConfig
from .dynamic_ai import DynamicAIManager
from .model_wrapper import ModelWrapper
from typing import List, Dict, Any, Optional, Set, Union, Tuple

logger: logging.Logger = logging.getLogger("logger")

class Pipeline:
    def __init__(self, config: PipelineConfig, model_manager: ModelManager, category_config: Dict, dynamic_ai_manager: DynamicAIManager):
        self.short_name: str = config.short_name
        self.version: float = config.version
        self.inputs: List[str] = config.inputs
        self.output: str = config.output
        self.models: List[ModelWrapper] = []
        self.category_config = category_config

        for model_config in config.models:
            modelName: str = model_config.name
            model_inputs: List[str] = model_config.inputs
            model_outputs: Union[str, List[str]] = model_config.outputs
            
            if modelName == "dynamic_video_ai":
                dynamic_models: List[ModelWrapper] = dynamic_ai_manager.get_dynamic_video_ai_models(model_inputs, model_outputs if isinstance(model_outputs, list) else [model_outputs] if model_outputs else [])
                self.models.extend(dynamic_models)
                continue
            
            returned_model: Any = model_manager.get_or_create_model(modelName)
            self.models.append(ModelWrapper(returned_model, model_inputs, model_outputs, model_name_for_logging=modelName))

        categories_set: Set[str] = set()
        for wrapper_model in self.models:
            if hasattr(wrapper_model.model, 'model') and isinstance(wrapper_model.model.model, AIModel):
                current_categories: Union[str, List[str], None] = wrapper_model.model.model.model_category
                if isinstance(current_categories, str):
                    if current_categories in categories_set:
                        raise ValueError(f"Error: AI models must not have overlapping categories! Category: {current_categories}")
                    categories_set.add(current_categories)
                elif isinstance(current_categories, list):
                    for cat in current_categories:
                        if cat in categories_set:
                            raise ValueError(f"Error: AI models must not have overlapping categories! Category: {cat}")
                        categories_set.add(cat)
        
        is_vlm_pipeline: bool = any(isinstance(mw.model.model, VLMAIModel) for mw in self.models if hasattr(mw.model, 'model'))
        if is_vlm_pipeline:
            for model_wrapper in self.models:
                if hasattr(model_wrapper.model, 'model') and isinstance(model_wrapper.model.model, VideoPreprocessorModel):
                    model_wrapper.model.model.set_vlm_pipeline_mode(True)
    
    async def event_handler(self, itemFuture: ItemFuture, key: str) -> None:
        if key == self.output:
            if key in itemFuture:
                itemFuture.close_future(itemFuture[key])
            else:
                pass
        
        for current_model_wrapper in self.models:
            if key in current_model_wrapper.inputs:
                allOtherInputsPresent: bool = True
                for inputName in current_model_wrapper.inputs:
                    if inputName != key:
                        is_present = (itemFuture.data is not None and inputName in itemFuture.data)
                        if not is_present:
                            allOtherInputsPresent = False
                            break
                
                if allOtherInputsPresent:
                    await current_model_wrapper.model.add_to_queue(QueueItem(itemFuture, current_model_wrapper.inputs, current_model_wrapper.outputs))

    async def start_model_processing(self) -> None:
        for model_wrapper in self.models:
            if hasattr(model_wrapper.model, 'start_workers') and callable(model_wrapper.model.start_workers):
                 await model_wrapper.model.start_workers()

    def get_ai_models_info(self) -> List[Tuple[Union[str, float, None], Optional[str], Optional[str], Optional[Union[str, List[str]]]]]:
        ai_version_and_ids: List[Tuple[Union[str, float, None], Optional[str], Optional[str], Optional[Union[str, List[str]]]]] = []
        for model_wrapper in self.models:
            if hasattr(model_wrapper.model, 'model') and isinstance(model_wrapper.model.model, AIModel):
                inner_ai_model: AIModel = model_wrapper.model.model
                version = inner_ai_model.model_version
                identifier = inner_ai_model.model_identifier
                file_name = inner_ai_model.model_file_name
                category = inner_ai_model.model_category
                ai_version_and_ids.append((version, identifier, file_name, category))
        return ai_version_and_ids

class PipelineManager:
    def __init__(self, pipelines_config: Dict[str, PipelineConfig], model_manager: ModelManager, category_config: Dict, dynamic_ai_manager: DynamicAIManager):
        self.pipelines: Dict[str, Pipeline] = {}
        self.logger: logging.Logger = logging.getLogger("logger")
        self.model_manager = model_manager
        self.pipelines_config = pipelines_config
        self.category_config = category_config
        self.dynamic_ai_manager = dynamic_ai_manager
    
    async def load_pipelines(self):
        for pipeline_name, pipeline_config in self.pipelines_config.items():
            self.logger.info(f"Loading pipeline: {pipeline_name}")
            try:
                new_pipeline = Pipeline(pipeline_config, self.model_manager, self.category_config, self.dynamic_ai_manager)
                self.pipelines[pipeline_name] = new_pipeline
                await new_pipeline.start_model_processing()
                self.logger.info(f"Pipeline {pipeline_name} V{new_pipeline.version} loaded successfully!")
            except Exception as e:
                if pipeline_name in self.pipelines:
                    del self.pipelines[pipeline_name]
                self.logger.error(f"Error loading pipeline {pipeline_name}: {e}")
                self.logger.debug("Exception details:", exc_info=True)
            
        if not self.pipelines:
            raise Exception("Error: No valid pipelines loaded!")
            
    def get_pipeline(self, pipeline_name: str) -> Pipeline:
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not found.")
        return self.pipelines[pipeline_name]

    async def get_request_future(self, data: List[Any], pipeline_name: str) -> ItemFuture:
        pipeline = self.get_pipeline(pipeline_name)
        futureData: Dict[str, Any] = {}
        if len(data) != len(pipeline.inputs):
            raise ValueError(f"Error: Data length does not match pipeline inputs length for pipeline {pipeline_name}!")
        
        for inputName, inputData in zip(pipeline.inputs, data):
            futureData[inputName] = inputData
        futureData["pipeline"] = pipeline
        futureData["category_config"] = self.category_config
        itemFuture: ItemFuture = await ItemFuture.create(None, futureData, pipeline.event_handler)
        return itemFuture
