from typing import List, Any, Union

# Forward reference for ModelProcessor if needed, or use Any
# from ..async_processing.async_processor import ModelProcessor 

class ModelWrapper:
    def __init__(self, 
                 model_processor: Any, # Should be ModelProcessor instance from async_processing
                 inputs: List[str], 
                 outputs: Union[str, List[str]], 
                 model_name_for_logging: str = "UnknownModel"):
        """
        Wraps a model processor along with its expected input and output names 
        within a pipeline.

        Args:
            model_processor: The ModelProcessor instance that handles the actual model execution.
            inputs: A list of string keys identifying the data inputs this model expects
                    from the pipeline's shared data context (ItemFuture.data).
            outputs: A string or list of string keys identifying the data outputs this model
                     will produce and set in the pipeline's shared data context.
            model_name_for_logging: A descriptive name for the model, used in logging.
        """
        self.model_processor: Any = model_processor # Renamed from 'model' to 'model_processor' for clarity
        self.inputs: List[str] = inputs
        self.outputs: Union[str, List[str]] = outputs 
        self.model_name_for_logging: str = model_name_for_logging

    def __repr__(self) -> str:
        return (f"ModelWrapper(model_name='{self.model_name_for_logging}', "
                f"inputs={self.inputs}, outputs={self.outputs})")
