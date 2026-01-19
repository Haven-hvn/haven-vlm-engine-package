from typing import List, Any, Union
import logging
import traceback

logger = logging.getLogger(__name__)

class ModelWrapper:
    def __init__(self, model: Any, inputs: List[str], outputs: Union[str, List[str]], model_name_for_logging: str = "UnknownModel"):
        self.model: Any = model
        self.inputs: List[str] = inputs
        self.outputs: Union[str, List[str]] = outputs 
        self.model_name_for_logging: str = model_name_for_logging
        
        # Debug instrumentation for exit status 120 investigation
        logger.debug(f"[DEBUG_MODEL_WRAPPER] ModelWrapper initialized for {model_name_for_logging}")

