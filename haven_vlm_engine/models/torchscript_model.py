import gc
import torch
import logging

logger = logging.getLogger("logger")

class TorchScriptModel: # Renamed from PythonModel
    def __init__(self, model_path: str, batch_size: int, device_str: Optional[str], fill_to_batch_size: bool, instance_name: Optional[str] = "TorchScriptModel"):
        self.model_path = model_path
        self.max_batch_size = batch_size
        self.fill_to_batch_size = fill_to_batch_size
        self.instance_name = instance_name # For logging
        
        if device_str:
            self.device = torch.device(device_str)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._model_loaded = False
        self.model: Optional[torch.jit.ScriptModule] = None
        # self.load_model() # Loading should be explicit via model_processor.load()

    def run_model(self, preprocessed_images: torch.Tensor, apply_sigmoid: bool) -> torch.Tensor:
        if not self._model_loaded or self.model is None:
            logger.error(f"TorchScriptModel '{self.instance_name}' not loaded. Call load_model() first.")
            raise RuntimeError(f"Model '{self.instance_name}' is not loaded.")
        
        preprocessed_images = preprocessed_images.to(self.device)
        original_batch_size = preprocessed_images.size(0)
        
        if self.fill_to_batch_size and original_batch_size < self.max_batch_size:
            if original_batch_size == 0: # Handle empty tensor case
                return torch.empty(0, device=self.device) # Or match expected output shape with 0 batch
            padding_size = self.max_batch_size - original_batch_size
            padding = torch.zeros((padding_size, *preprocessed_images.shape[1:]), dtype=preprocessed_images.dtype, device=self.device)
            preprocessed_images = torch.cat([preprocessed_images, padding], dim=0)
        
        if preprocessed_images.size(0) > self.max_batch_size:
            raise ValueError(f"Batch size {preprocessed_images.size(0)} for '{self.instance_name}' exceeds model's max_batch_size {self.max_batch_size}")

        # Half precision conversion
        model_is_half = False
        if self.device.type == 'cuda':
            try:
                # Check if model parameters are already in half precision
                if self.model and next(self.model.parameters()).dtype == torch.float16:
                    model_is_half = True
            except StopIteration: # Model has no parameters
                pass 
            
            if preprocessed_images.dtype == torch.float32 and not model_is_half:
                 # Only convert input to half if model is not already half, to prevent mismatch
                 # This assumes model can handle half if input is half.
                preprocessed_images = preprocessed_images.half()


        with torch.no_grad():
            if self.device.type == 'cuda':
                # Enable autocast if input or model is half, or if desired for performance
                # Using dtype of preprocessed_images for autocast context
                autocast_dtype = torch.float16 if preprocessed_images.dtype == torch.float16 else torch.float32
                with torch.autocast(device_type=self.device.type, enabled=True, dtype=autocast_dtype):
                    output = self.model(preprocessed_images)
            else: # CPU
                output = self.model(preprocessed_images)
            
            if apply_sigmoid:
                output = torch.sigmoid(output)
    
        output = output[:original_batch_size]
        return output.cpu()

    def process_images(self, preprocessed_images: torch.Tensor, apply_sigmoid: bool = True) -> torch.Tensor:
        if not self._model_loaded or self.model is None:
            logger.error(f"TorchScriptModel '{self.instance_name}' not loaded. Cannot process images.")
            raise RuntimeError(f"Model '{self.instance_name}' is not loaded.")
        
        if preprocessed_images.size(0) == 0:
            return torch.empty(0, device=torch.device('cpu'))

        if preprocessed_images.size(0) <= self.max_batch_size:
            return self.run_model(preprocessed_images, apply_sigmoid)
        else:
            chunks = torch.split(preprocessed_images, self.max_batch_size)
            results = []
            for chunk in chunks:
                chunk_result = self.run_model(chunk, apply_sigmoid)
                results.append(chunk_result)
            return torch.cat(results, dim=0)

    def load_model(self) -> None:
        if self._model_loaded:
            logger.debug(f"TorchScriptModel '{self.instance_name}' already loaded.")
            return
        logger.info(f"Loading TorchScriptModel '{self.instance_name}' from {self.model_path} to {self.device}...")
        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval() # Set to evaluation mode
            self._model_loaded = True
            logger.info(f"TorchScriptModel '{self.instance_name}' loaded successfully.")
        except Exception as e:
            self._model_loaded = False
            logger.error(f"Failed to load TorchScriptModel '{self.instance_name}' from {self.model_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.instance_name}' from {self.model_path}: {e}") from e

    def unload_model(self) -> None:
        logger.info(f"Unloading TorchScriptModel '{self.instance_name}'...")
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self._model_loaded = False
        logger.info(f"TorchScriptModel '{self.instance_name}' unloaded.")

    @property
    def model_loaded(self) -> bool: # Make it a property
        return self._model_loaded
        
    # __enter__ and __exit__ can be useful if the model is used with a 'with' statement
    # but typically loading/unloading is managed by the ModelProcessor's lifecycle.
    # For now, let's keep them simple if they are to be kept.
    def __enter__(self):
        if not self.model_loaded:
            self.load_model()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # self.unload_model() # Unloading on exit might be too aggressive if instance is reused.
        pass # Let ModelProcessor manage unload via its lifecycle or explicit call.
