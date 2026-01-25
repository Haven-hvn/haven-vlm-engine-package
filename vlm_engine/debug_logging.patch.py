"""Debug logging patch to add more detailed logging to models.py to track the actual error"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("vlm_engine_debug")

# Let's patch the PythonModel.__init__ to log more details
original_python_model_init = None

def debug_python_model_init(self, configValues):
    """Patched __init__ for PythonModel to log debug info"""
    logger.debug(f"[DEBUG_PYTHON_MODEL] PythonModel.__init__ called with configValues type: {type(configValues)}")
    logger.debug(f"[DEBUG_PYTHON_MODEL] configValues attributes: {dir(configValues)}")
    
    # Log the function_name specifically
    function_name_attr = getattr(configValues, 'function_name', 'NOT_SET_ATTR')
    logger.debug(f"[DEBUG_PYTHON_MODEL] configValues.function_name attribute: {repr(function_name_attr)}")
    
    # Try to get dict representation
    try:
        config_dict = configValues.dict()
        logger.debug(f"[DEBUG_PYTHON_MODEL] configValues.dict(): {config_dict}")
        logger.debug(f"[DEBUG_PYTHON_MODEL] Full config for debugging: {config_dict}")
    except Exception as e:
        logger.debug(f"[DEBUG_PYTHON_MODEL] Could not get dict: {e}")
    
    # Call original __init__
    return original_python_model_init(self, configValues)

# Also patch ModelManager.model_factory to see which model is being created
original_model_factory = None

def debug_model_factory(self, model_config):
    """Patched model_factory to log which model is being created"""
    logger.debug(f"[DEBUG_MODEL_FACTORY] Creating model with config type: {model_config.type}")
    logger.debug(f"[DEBUG_MODEL_FACTORY] Model config dict: {model_config.dict()}")
    
    if model_config.type == 'python':
        logger.debug(f"[DEBUG_MODEL_FACTORY] Python model detected! function_name: {repr(model_config.function_name)}")
        if model_config.function_name is None:
            logger.error(f"[DEBUG_MODEL_FACTORY] ⚠️ WARNING: Python model with None function_name!")
        elif model_config.function_name == "":
            logger.error(f"[DEBUG_MODEL_FACTORY] ⚠️ WARNING: Python model with empty string function_name!")
    
    return original_model_factory(self, model_config)

# Patch ModelManager.create_model to see which model name is being created
original_create_model = None

def debug_create_model(self, modelName):
    """Patched create_model to log model name"""
    logger.debug(f"[DEBUG_CREATE_MODEL] Creating model: {modelName}")
    if modelName in self.models_config:
        config = self.models_config[modelName]
        logger.debug(f"[DEBUG_CREATE_MODEL] Model config for {modelName}: type={config.type}, function_name={repr(getattr(config, 'function_name', 'NO_ATTR'))}")
    else:
        logger.error(f"[DEBUG_CREATE_MODEL] Model {modelName} not found in models_config!")
    
    return original_create_model(self, modelName)

# Apply patches
def apply_debug_patches():
    """Apply all debug patches to models.py"""
    import vlm_engine.models
    import vlm_engine.models as models_module
    
    global original_python_model_init, original_model_factory, original_create_model
    
    # Patch PythonModel.__init__
    original_python_model_init = models_module.PythonModel.__init__
    models_module.PythonModel.__init__ = debug_python_model_init
    logger.debug("[DEBUG] Patched PythonModel.__init__")
    
    # Patch ModelManager.model_factory
    original_model_factory = models_module.ModelManager.model_factory
    models_module.ModelManager.model_factory = debug_model_factory
    logger.debug("[DEBUG] Patched ModelManager.model_factory")
    
    # Patch ModelManager.create_model
    original_create_model = models_module.ModelManager.create_model
    models_module.ModelManager.create_model = debug_create_model
    logger.debug("[DEBUG] Patched ModelManager.create_model")
    
    return True

# Test the patches
if __name__ == "__main__":
    print("Testing debug logging patches...")
    apply_debug_patches()
    
    # Test with a problematic config
    from vlm_engine.config_models import ModelConfig
    
    print("\n1. Testing with valid python model config:")
    try:
        config1 = ModelConfig(type="python", function_name="test_func")
        model1 = models_module.PythonModel(config1)
        print(f"✓ Valid model created")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n2. Testing with python model config with NO function_name:")
    try:
        config2 = ModelConfig(type="python")  # Missing function_name
        model2 = models_module.PythonModel(config2)
        print(f"✓ Model created (should have failed?)")
    except Exception as e:
        print(f"✗ Expected error: {e}")