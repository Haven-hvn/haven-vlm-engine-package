from typing import List, Union
from .item_future import ItemFuture # Updated import

class QueueItem:
    def __init__(self, itemFuture: ItemFuture, input_names: List[str], output_names: Union[str, List[str]]):
        self.item_future: ItemFuture = itemFuture
        self.input_names: List[str] = input_names
        self.output_names: Union[str, List[str]] = output_names
