import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from .models import Model, AIModel

logger: logging.Logger = logging.getLogger("logger")

class ItemFuture:
    def __init__(self, parent: Optional['ItemFuture'], event_handler: Callable[['ItemFuture', str], Awaitable[None]]):
        self.parent: Optional['ItemFuture'] = parent
        self.handler: Callable[['ItemFuture', str], Awaitable[None]] = event_handler
        self.future: asyncio.Future[Any] = asyncio.Future()
        self.data: Optional[Dict[str, Any]] = {}

    async def set_data(self, key: str, value: Any) -> None:
        if self.data is not None:
            self.data[key] = value
        await self.handler(self, key)

    async def __setitem__(self, key: str, value: Any) -> None:
        await self.set_data(key, value)

    def close_future(self, value: Any) -> None:
        self.data = None
        if not self.future.done():
            self.future.set_result(value)

    def set_exception(self, exception: Exception) -> None:
        self.data = None
        if not self.future.done():
            self.future.set_exception(exception)

    def __contains__(self, key: str) -> bool:
        if self.data is None:
            return False
        is_present = key in self.data
        return is_present

    def __getitem__(self, key: str) -> Any:
        if self.data is None:
            return None
        
        value = self.data.get(key)
        return value

    def __await__(self) -> Generator[Any, None, Any]:
        yield from self.future.__await__()
        return self.future.result()

    @classmethod
    async def create(cls, parent: Optional['ItemFuture'], data: Dict[str, Any], event_handler: Callable[['ItemFuture', str], Awaitable[None]]) -> 'ItemFuture':
        self_ref: 'ItemFuture' = cls(parent, event_handler)
        if self_ref.data is not None:
            key: str
            for key in data:
                self_ref.data[key] = data[key]
                await self_ref.handler(self_ref, key)
        return self_ref

class QueueItem:
    def __init__(self, itemFuture: ItemFuture, input_names: List[str], output_names: Union[str, List[str]]):
        self.item_future: ItemFuture = itemFuture
        self.input_names: List[str] = input_names
        self.output_names: Union[str, List[str]] = output_names

class ModelProcessor():
    def __init__(self, model: 'Model'):
        self.model: 'Model' = model
        self.instance_count: int = model.instance_count
        if model.max_queue_size is None:
            self.queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        else:
            self.queue: asyncio.Queue[QueueItem] = asyncio.Queue(maxsize=model.max_queue_size)
        self.max_batch_size: int = self.model.max_batch_size
        self.workers_started: bool = False
        self.failed_loading: bool = False
        from .models import AIModel
        self.is_ai_model: bool = isinstance(self.model, AIModel)
        
    async def add_to_queue(self, data: QueueItem) -> None:
        await self.queue.put(data)

    async def worker_process(self) -> None:
        model_identifier = getattr(self.model, 'model_identifier', 'UnknownModel')
        while True:
            try:
                firstItem: QueueItem = await self.queue.get()
                
                batch_data: List[QueueItem] = [firstItem]
                
                while len(batch_data) < self.max_batch_size:
                    if not self.queue.empty():
                        next_item: QueueItem = await self.queue.get()
                        batch_data.append(next_item)
                    else:
                        break
                
                if batch_data:
                    try:
                        await self.model.worker_function_wrapper(batch_data)
                    except Exception as e:
                        for item_in_batch in batch_data:
                            if not item_in_batch.item_future.done():
                                try:
                                    item_in_batch.item_future.set_exception(e)
                                except Exception as fut_e:
                                    logger.error(f"Exception setting exception on ItemFuture: {fut_e}")
                    finally:
                        for _ in batch_data:
                            self.queue.task_done()
            except Exception as e:
                self.failed_loading = True
                logger.error(f"Failed to start workers for model '{model_identifier}': {e}", exc_info=True)
                raise

    async def start_workers(self) -> None:
        if self.workers_started:
            if self.failed_loading:
                raise Exception("Error: Model failed to load previously!") 
            return
        try:
            self.workers_started = True
            await self.model.load()
            for _ in range(self.instance_count):
                asyncio.create_task(self.worker_process())
        except Exception as e:
            self.failed_loading = True
            logger.error(f"Failed to start workers for model: {e}", exc_info=True)
            raise
