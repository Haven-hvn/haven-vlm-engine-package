class EngineException(Exception):
    """Base class for exceptions raised by the VLM engine package."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class PipelineException(EngineException):
    """Raised for errors during pipeline configuration or execution."""
    pass

class ModelException(EngineException):
    """Raised for errors related to model loading or processing."""
    pass

class ConfigurationException(EngineException):
    """Raised for errors in package configuration."""
    pass

class NoActiveModelsException(ModelException):
    """Raised when no active models are specified or found for a dynamic group."""
    pass

class ModelNotFoundException(ModelException):
    """Raised when a specified model instance name cannot be found or loaded."""
    def __init__(self, model_name: str, message: str = ""):
        self.model_name = model_name
        if not message:
            message = f"Model instance '{model_name}' not found or could not be loaded."
        super().__init__(message)
