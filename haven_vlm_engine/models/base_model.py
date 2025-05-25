import logging
from typing import Dict, Any, Optional, List
from ..async_processing.queue_item import QueueItem # Updated import path


class Model:
    def __init__(self, configValues: Dict[str, Any]):
        self.max_queue_size: Optional[int] = configValues.get("max_queue_size")
        self.max_batch_size: int = int(configValues.get("max_batch_size", 1))
        self.instance_count: int = int(configValues.get("instance_count", 1))
        self.max_batch_waits: int = int(configValues.get("max_batch_waits", -1))
        self.logger: logging.Logger = logging.getLogger("logger") # This will need to be configured by the package user

    async def worker_function_wrapper(self, data: List[QueueItem]) -> None:
        try:
            await self.worker_function(data)
        except Exception as e:
            self.logger.error(f"Exception in worker_function: {e}", exc_info=True)
            item: QueueItem
            for item in data:
                if hasattr(item, 'item_future') and item.item_future:
                    item.item_future.set_exception(e)
                else:
                    self.logger.error("Item in batch lacks item_future, cannot propagate exception.")

    async def worker_function(self, data: List[QueueItem]) -> None:
        # This method is intended to be overridden by subclasses
        pass

    async def load(self) -> None:
        # This method is intended to be overridden by subclasses
        return
