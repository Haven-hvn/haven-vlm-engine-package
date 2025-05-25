from typing import List, Union, Optional, Dict, Any, Callable, Awaitable, Generator
import asyncio
import logging

logger: logging.Logger = logging.getLogger("logger")

class ItemFuture:
    def __init__(self, parent: Optional['ItemFuture'], event_handler: Callable[['ItemFuture', str], Awaitable[None]]):
        self.parent: Optional['ItemFuture'] = parent
        self.handler: Callable[['ItemFuture', str], Awaitable[None]] = event_handler
        self.future: asyncio.Future[Any] = asyncio.Future()
        self.data: Optional[Dict[str, Any]] = {} # Holds the actual data items

    async def set_data(self, key: str, value: Any) -> None:
        if self.data is not None:
            self.data[key] = value
        # Call the event handler which might trigger further processing or future completion
        await self.handler(self, key)

    async def __setitem__(self, key: str, value: Any) -> None:
        # Syntactic sugar for self.set_data
        await self.set_data(key, value)

    def close_future(self, value: Any) -> None:
        """
        Marks the underlying asyncio.Future as done with a result.
        Data dictionary is cleared.
        """
        if not self.future.done():
            self.future.set_result(value)
        self.data = None # Clear data once future is resolved

    def set_exception(self, exception: Exception) -> None:
        """
        Marks the underlying asyncio.Future as done with an exception.
        Data dictionary is cleared.
        """
        if not self.future.done():
            self.future.set_exception(exception)
        self.data = None # Clear data on exception

    def __contains__(self, key: str) -> bool:
        # Check if a key is in the data dictionary
        return self.data is not None and key in self.data

    def __getitem__(self, key: str) -> Any:
        # Retrieve an item from the data dictionary
        if self.data is None:
            # Or raise KeyError, depending on desired behavior for closed/errored future
            logger.warning(f"Attempted to get item '{key}' from a closed or errored ItemFuture.")
            return None 
        return self.data.get(key) # .get() returns None if key not found, vs KeyError

    def get(self, key: str, default: Any = None) -> Any:
        """Provides a .get() method similar to dictionaries."""
        if self.data is None:
            return default
        return self.data.get(key, default)

    def done(self) -> bool:
        """Checks if the underlying asyncio.Future is done."""
        return self.future.done()

    def __await__(self) -> Generator[Any, None, Any]:
        # Allows 'await item_future_instance'
        yield from self.future.__await__()
        return self.future.result() # Raise exception if future has one

    @classmethod
    async def create(cls, parent: Optional['ItemFuture'], data: Dict[str, Any], event_handler: Callable[['ItemFuture', str], Awaitable[None]]) -> 'ItemFuture':
        """
        Factory method to create and initialize an ItemFuture.
        Initial data items are set, triggering the event handler for each.
        """
        self_ref: 'ItemFuture' = cls(parent, event_handler)
        if self_ref.data is not None: # Should always be true after __init__
            for key, value in data.items(): # Iterate over items directly
                # Set data first, then call handler, as set_data does.
                self_ref.data[key] = value 
                await self_ref.handler(self_ref, key)
        return self_ref
