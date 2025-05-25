import asyncio
import importlib
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable

from ..async_processing.queue_item import QueueItem
from .base_model import Model as BaseModel # Alias to avoid clash if BaseModel is also imported directly
from ..utils.exceptions import ConfigurationException, ModelException

logger: logging.Logger = logging.getLogger("logger")

class PythonCallableModel(BaseModel):
    def __init__(self, model_config: Dict[str, Any]):
        """
        A model that executes a specified Python async callable.

        Expected model_config keys:
        - module_path: (str) The dot-separated path to the module (e.g., "my_package.my_module").
        - function_name: (str) The name of the async function within the module.
        - instance_name: (Optional[str]) Name for logging.
        - max_queue_size, max_batch_size, instance_count, max_batch_waits (from BaseModel)
        """
        super().__init__(model_config) # Initializes BaseModel attributes like logger, batching params
        
        self.module_path: Optional[str] = model_config.get("module_path")
        self.function_name: Optional[str] = model_config.get("function_name")
        self.instance_name: str = model_config.get("instance_name", f"{self.module_path}.{self.function_name}")

        if not self.module_path or not self.function_name:
            raise ConfigurationException(
                f"PythonCallableModel '{self.instance_name}' requires 'module_path' and 'function_name' in its configuration."
            )

        self._callable_fn: Optional[Callable[[List[QueueItem]], Awaitable[None]]] = None
        self._loaded = False # To track if the callable has been loaded

    async def load(self) -> None:
        if self._loaded:
            logger.debug(f"PythonCallableModel '{self.instance_name}' already loaded.")
            return
        
        logger.info(f"Loading PythonCallableModel '{self.instance_name}': {self.module_path}.{self.function_name}")
        try:
            module = importlib.import_module(self.module_path)
            func = getattr(module, self.function_name)
            
            if not callable(func):
                raise TypeError(f"Attribute '{self.function_name}' in module '{self.module_path}' is not callable.")
            if not asyncio.iscoroutinefunction(func):
                raise TypeError(f"Function '{self.function_name}' in module '{self.module_path}' must be an async function (coroutine).")
                
            self._callable_fn = func
            self._loaded = True
            logger.info(f"PythonCallableModel '{self.instance_name}' loaded successfully.")
        except ImportError:
            logger.error(f"Failed to import module '{self.module_path}' for PythonCallableModel '{self.instance_name}'.", exc_info=True)
            raise ConfigurationException(f"Could not import module: {self.module_path}")
        except AttributeError:
            logger.error(f"Failed to find function '{self.function_name}' in module '{self.module_path}' for PythonCallableModel '{self.instance_name}'.", exc_info=True)
            raise ConfigurationException(f"Could not find function '{self.function_name}' in module '{self.module_path}'.")
        except TypeError as te: # Catch specific type errors from checks
            logger.error(f"Type error for PythonCallableModel '{self.instance_name}': {te}", exc_info=True)
            raise ConfigurationException(str(te))
        except Exception as e:
            logger.error(f"Unexpected error loading PythonCallableModel '{self.instance_name}': {e}", exc_info=True)
            raise ModelException(f"Failed to load callable for '{self.instance_name}': {e}")


    async def worker_function(self, queue_items: List[QueueItem]) -> None:
        """
        This is the function called by ModelProcessor with a batch of QueueItems.
        It directly calls the configured async Python callable.
        """
        if not self._loaded or self._callable_fn is None:
            err_msg = f"PythonCallableModel '{self.instance_name}' is not loaded. Cannot process."
            logger.error(err_msg)
            # Set exception for all items in the batch
            for item in queue_items:
                if item.item_future and not item.item_future.done():
                    item.item_future.set_exception(ModelException(err_msg))
            return # Or raise, but setting exception on futures is cleaner for batch

        try:
            # The callable is expected to handle the list of QueueItems directly
            # and manage their ItemFutures (setting data or exceptions).
            await self._callable_fn(queue_items)
        except Exception as e:
            logger.error(f"Error during execution of PythonCallableModel '{self.instance_name}' ({self.module_path}.{self.function_name}): {e}", exc_info=True)
            # Propagate the exception to all item_futures in the batch
            for item in queue_items:
                if item.item_future and not item.item_future.done():
                    item.item_future.set_exception(e)
            # Optionally re-raise if the ModelProcessor should also handle this
            # raise ModelProcessingException(f"Execution failed in {self.instance_name}") from e

    # worker_function_wrapper is inherited from BaseModel and calls this worker_function.
