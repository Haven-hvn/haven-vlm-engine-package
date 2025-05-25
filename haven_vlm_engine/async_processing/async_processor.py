import asyncio
import logging
from typing import List, Any, Optional, Union, TYPE_CHECKING

# Updated imports for the new package structure
from ..utils.skip_input import Skip, SKIP_INSTANCE # Assuming SKIP_INSTANCE will be used
from .queue_item import QueueItem 

if TYPE_CHECKING:
    from ..models.base_model import Model # For type hinting ModelProcessor.model
    from ..models.ai_model import AIModel # For type hinting ModelProcessor.is_ai_model check
else:
    Model = Any 
    AIModel = Any # Runtime placeholders if not type checking

logger: logging.Logger = logging.getLogger("logger")

class ModelProcessor():
    def __init__(self, model: 'Model'):
        self.model: 'Model' = model
        self.instance_count: int = model.instance_count
        if model.max_queue_size is None:
            self.queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        else:
            self.queue: asyncio.Queue[QueueItem] = asyncio.Queue(maxsize=model.max_queue_size)
        
        self.max_batch_size: int = self.model.max_batch_size
        self.max_batch_waits: int = self.model.max_batch_waits
        self.workers_started: bool = False
        self.failed_loading: bool = False
        
        # Dynamically import AIModel for isinstance check to avoid circular dependency at module level
        # This is a common pattern if AIModel itself might import things from async_processing indirectly.
        from ..models.ai_model import AIModel as ConcreteAIModel 
        self.is_ai_model: bool = isinstance(self.model, ConcreteAIModel)

    def update_values_from_child_model(self) -> None:
        """Updates processor's queue and batch settings from its model instance."""
        self.instance_count = self.model.instance_count
        if self.model.max_queue_size is None:
            self.queue = asyncio.Queue()
        else:
            self.queue = asyncio.Queue(maxsize=self.model.max_queue_size)
        self.max_batch_size = self.model.max_batch_size
        self.max_batch_waits = self.model.max_batch_waits
        
    async def add_to_queue(self, data: QueueItem) -> None:
        await self.queue.put(data)

    async def add_items_to_queue(self, data: List[QueueItem]) -> None:
        for item in data:
            await self.queue.put(item)

    async def complete_item_with_skip(self, item: QueueItem) -> None:
        """Completes an item by setting its output(s) to SKIP_INSTANCE."""
        if isinstance(item.output_names, list):
            for output_target in item.output_names:
                # Using SKIP_INSTANCE from the imported module
                await item.item_future.set_data(output_target, SKIP_INSTANCE)
        else: # it's a string
            await item.item_future.set_data(item.output_names, SKIP_INSTANCE)

    async def batch_data_append_with_skips(self, batch_data: List[QueueItem], item: QueueItem) -> bool:
        """
        Appends item to batch_data if it's not skipped.
        If skipped, completes the item and returns True. Otherwise, returns False.
        """
        if self.is_ai_model:
            from ..models.ai_model import AIModel as ConcreteAIModel # Ensure AIModel is in scope
            # We need to cast self.model to AIModel to access model_category
            # This assumes self.is_ai_model check in __init__ is reliable.
            ai_model_instance = self.model
            if isinstance(ai_model_instance, ConcreteAIModel):
                # Input names for skipped_categories should be standardized or configurable.
                # Assuming item.input_names[3] is 'skipped_categories' by convention.
                skipped_categories_input_name = item.input_names[3] if len(item.input_names) > 3 else None
                
                skipped_categories: Optional[List[str]] = None
                if skipped_categories_input_name and skipped_categories_input_name in item.item_future:
                    skipped_categories = item.item_future[skipped_categories_input_name]

                if skipped_categories is not None:
                    this_ai_categories: Optional[Union[str, List[str]]] = ai_model_instance.model_category
                    if this_ai_categories:
                        is_skipped = False
                        if isinstance(this_ai_categories, str):
                            if this_ai_categories in skipped_categories:
                                is_skipped = True
                        elif isinstance(this_ai_categories, list):
                            if all(cat in skipped_categories for cat in this_ai_categories):
                                is_skipped = True
                        
                        if is_skipped:
                            await self.complete_item_with_skip(item)
                            return True
        batch_data.append(item)
        return False

    async def worker_process(self) -> None:
        model_identifier = getattr(self.model, 'model_identifier', getattr(self.model, 'function_name', 'UnknownModel'))
        while True:
            try:
                firstItem: QueueItem = await self.queue.get()
                
                batch_data: List[QueueItem] = []
                if await self.batch_data_append_with_skips(batch_data, firstItem):
                    self.queue.task_done()
                    continue

                waitsSoFar: int = 0
                while len(batch_data) < self.max_batch_size and \
                      (self.max_batch_waits == -1 or waitsSoFar < self.max_batch_waits):
                    try:
                        # Non-blocking get if we are just filling up to max_batch_size quickly
                        # or if max_batch_waits is 0 (meaning process immediately if batch not full)
                        timeout_for_get = 0.001 if self.max_batch_waits == 0 and len(batch_data) > 0 else 1.0
                        next_item: QueueItem = await asyncio.wait_for(self.queue.get(), timeout=timeout_for_get if self.max_batch_waits != -1 else None)
                        if await self.batch_data_append_with_skips(batch_data, next_item):
                            self.queue.task_done()
                    except asyncio.TimeoutError:
                        if self.max_batch_waits != -1: # If not waiting indefinitely
                            waitsSoFar += 1
                            if waitsSoFar >= self.max_batch_waits : break # Exhausted waits
                        # If max_batch_waits is -1, timeout shouldn't occur unless None was passed to wait_for.
                        # If max_batch_waits is 0, we break immediately after one timeout if queue was empty.
                        if self.max_batch_waits == 0 : break 
                    except asyncio.QueueEmpty: # Should not happen with await self.queue.get() unless timeout
                        if self.max_batch_waits != -1: waitsSoFar +=1 
                        if self.max_batch_waits == 0 or (self.max_batch_waits != -1 and waitsSoFar >= self.max_batch_waits): break
                
                if batch_data:
                    try:
                        await self.model.worker_function_wrapper(batch_data)
                    except Exception as e_process:
                        logger.error(f"Error during model processing for {model_identifier}: {e_process}", exc_info=True)
                        for item_in_batch in batch_data:
                            if not item_in_batch.item_future.done():
                                item_in_batch.item_future.set_exception(e_process)
                    finally:
                        for _ in batch_data:
                            self.queue.task_done()
                else: # Case where firstItem was fetched but loop for more items didn't run (e.g. max_batch_size=1)
                    # This 'else' might be redundant if firstItem is always processed if batch_data is empty.
                    # However, if firstItem was the *only* item and batch_data_append_with_skips returned false,
                    # it's in batch_data. If it was skipped, task_done was already called.
                    # This path (empty batch_data after trying to populate) should be rare.
                    # If firstItem was valid and added, batch_data won't be empty.
                    # If firstItem was skipped, we 'continue'd.
                    # So, if batch_data is empty here, it means queue was empty after firstItem.
                    # We still need to mark firstItem as done if it wasn't skipped.
                    # This logic seems complex; the original didn't have this specific 'else'.
                    # The original `if batch_data:` implicitly handles the case where only firstItem is processed.
                    # If firstItem was valid, batch_data = [firstItem].
                    # If firstItem was skipped, we continued.
                    # If queue became empty after firstItem, batch_data = [firstItem].
                    # Let's stick to simpler: if batch_data is populated, process it.
                    # The first self.queue.task_done() is only for the *skipped* firstItem.
                    # If firstItem is not skipped, it's part of batch_data and its task_done is in the finally block.
                    pass # No items in batch_data to process, firstItem was handled or queue was empty.


            except Exception as e_outer: # Catch errors in the main worker loop (e.g., queue.get errors)
                self.failed_loading = True # Potentially misleading, but indicates worker failure
                logger.error(f"Outer worker_process loop error for model '{model_identifier}': {e_outer}", exc_info=True)
                # Depending on error, may want to break or attempt to recover. For now, re-raise.
                # If queue.get() itself fails catastrophically, the loop might break.
                # Consider how to handle persistent queue errors.
                raise


    async def start_workers(self) -> None:
        if self.workers_started:
            if self.failed_loading:
                raise Exception(f"Model {getattr(self.model, 'model_identifier', 'UnknownModel')} failed to load previously or workers failed!") 
            return
        
        try:
            self.workers_started = True 
            await self.model.load() 
            for _ in range(self.instance_count):
                asyncio.create_task(self.worker_process())
        except Exception as e:
            self.failed_loading = True
            logger.error(f"Failed to start workers for model {getattr(self.model, 'model_identifier', 'UnknownModel')}: {e}", exc_info=True)
            raise
