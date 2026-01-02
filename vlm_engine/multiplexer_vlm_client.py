import asyncio
import logging
from typing import Dict, Any, Optional, List
from PIL import Image
import httpx
from multiplexer_llm import (
    Multiplexer,
    MultiplexerError,
    ModelNotFoundError,
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
    ModelSelectionError,
)
from openai import AsyncOpenAI
from .base_vlm_client import BaseVLMClient


class MultiplexerVLMClient(BaseVLMClient):
    """
    High-performance VLM client that uses multiplexer-llm for load balancing across multiple OpenAI-compatible endpoints.
    Features advanced async patterns, concurrency control, and optimized connection pooling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Performance optimization settings
        self.max_concurrent_requests: int = int(config.get("max_concurrent_requests", 20))
        self.connection_pool_size: int = int(config.get("connection_pool_size", 50))
        
        self.logger.debug(f"MultiplexerVLMClient initialized with {len(self.tag_list)} tags: {self.tag_list[:5]}...")
        
        # Extract multiplexer endpoints configuration
        self.multiplexer_endpoints: List[Dict[str, Any]] = config.get("multiplexer_endpoints", [])
        if not self.multiplexer_endpoints:
            raise ValueError("Configuration must provide 'multiplexer_endpoints' for multiplexer mode.")
        
        # Initialize concurrency control
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.multiplexer: Optional[Multiplexer] = None
        self._initialized = False
        
        self.logger.info(
            f"Initializing high-performance MultiplexerVLMClient for model {self.model_id} "
            f"with {len(self.tag_list)} tags, {len(self.multiplexer_endpoints)} endpoints, "
            f"max_concurrent: {self.max_concurrent_requests}, pool_size: {self.connection_pool_size}"
        )
    
    async def _ensure_initialized(self):
        """Ensure the multiplexer is initialized. Called before each request."""
        if not self._initialized:
            await self._initialize_multiplexer()
    
    async def _initialize_multiplexer(self):
        """Initialize the multiplexer with configured endpoints and optimized connection pooling."""
        if self._initialized:
            return
            
        self.logger.info("Initializing high-performance multiplexer with connection pooling...")
        
        # Configure HTTP client with optimized connection pooling
        limits = httpx.Limits(
            max_keepalive_connections=self.connection_pool_size,
            max_connections=self.connection_pool_size * 2,
            keepalive_expiry=30.0
        )
        
        # Create multiplexer instance
        self.multiplexer = Multiplexer()
        await self.multiplexer.__aenter__()
        
        # Add endpoints to multiplexer with optimized clients
        for i, endpoint_config in enumerate(self.multiplexer_endpoints):
            try:
                # Create optimized HTTP client for this endpoint
                http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.request_timeout),
                    limits=limits
                )
                
                # Create AsyncOpenAI client with optimized HTTP client
                client = AsyncOpenAI(
                    api_key=endpoint_config.get("api_key", "dummy_api_key"),
                    base_url=endpoint_config["base_url"],
                    http_client=http_client,
                    max_retries=2,
                    timeout=self.request_timeout
                )
                
                weight = endpoint_config.get("weight", 1)
                name = endpoint_config.get("name", f"endpoint-{i}")
                is_fallback = endpoint_config.get("is_fallback", False)
                
                if is_fallback:
                    self.multiplexer.add_fallback_model(client, weight, name)
                    self.logger.info(f"Added fallback endpoint: {name} (weight: {weight})")
                else:
                    self.multiplexer.add_model(client, weight, name)
                    self.logger.info(f"Added primary endpoint: {name} (weight: {weight})")
                    
            except Exception as e:
                self.logger.error(f"Failed to add endpoint {endpoint_config}: {e}")
                raise
        
        self._initialized = True
        self.logger.info(f"Multiplexer initialization completed with {len(self.multiplexer_endpoints)} endpoints and connection pooling")
    
    async def _cleanup_multiplexer(self):
        """Cleanup multiplexer resources."""
        if self.multiplexer and self._initialized:
            try:
                await self.multiplexer.__aexit__(None, None, None)
                self.logger.info("Multiplexer cleanup completed")
            except Exception as e:
                self.logger.error(f"Error during multiplexer cleanup: {e}")
            finally:
                self.multiplexer = None
                self._initialized = False
    
    async def analyze_frame(self, frame: Optional[Image.Image]) -> Dict[str, float]:
        """
        Analyze a frame using the multiplexer with concurrency control and proper exception handling.
        Features advanced async patterns and optimized performance.
        """
        if not frame:
            self.logger.warning("analyze_frame called with no frame.")
            return {tag: 0.0 for tag in self.tag_list}
        
        # Use semaphore for concurrency control
        async with self.semaphore:
            # Ensure multiplexer is initialized
            await self._ensure_initialized()
            
            try:
                image_data_url: str = self._convert_image_to_base64_data_url(frame)
            except Exception as e_convert:
                self.logger.error(f"Failed to convert image to base64: {e_convert}", exc_info=True)
                return {tag: 0.0 for tag in self.tag_list}
            
            prompt_text: str = self._build_prompt_text()
            
            messages: List[Dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {
                            "type": "text",
                            "text": prompt_text,
                        },
                    ],
                }
            ]
            
            try:
                # Use multiplexer for the request with proper exception handling
                completion = await self.multiplexer.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=0.0,
                    timeout=self.request_timeout
                )
                
                if completion.choices and completion.choices[0].message:
                    raw_reply = completion.choices[0].message.content or ""
                    self.logger.debug(f"Received response from multiplexer: {raw_reply[:100]}...")
                    return self._parse_simple_default(raw_reply)
                else:
                    self.logger.error(f"Unexpected response structure from multiplexer: {completion}")
                    return {tag: 0.0 for tag in self.tag_list}
                    
            except ModelNotFoundError as e:
                self.logger.error(f"Model not found at endpoint {e.endpoint}: {e.message}")
                return {tag: 0.0 for tag in self.tag_list}
                
            except AuthenticationError as e:
                self.logger.error(f"Authentication failed at endpoint {e.endpoint}: {e.message}")
                return {tag: 0.0 for tag in self.tag_list}
                
            except RateLimitError as e:
                self.logger.warning(f"Rate limit hit at endpoint {e.endpoint}, retry after {e.retry_after}s: {e.message}")
                return {tag: 0.0 for tag in self.tag_list}
                
            except ServiceUnavailableError as e:
                self.logger.error(f"Service unavailable at endpoint {e.endpoint}: {e.message}")
                return {tag: 0.0 for tag in self.tag_list}
                
            except ModelSelectionError as e:
                self.logger.error(f"No models available for selection: {e.message}")
                return {tag: 0.0 for tag in self.tag_list}
                
            except MultiplexerError as e:
                self.logger.error(f"Multiplexer error: {e.message}")
                return {tag: 0.0 for tag in self.tag_list}
                
            except Exception as e:
                self.logger.error(f"Unexpected error during frame analysis: {e}", exc_info=True)
                return {tag: 0.0 for tag in self.tag_list}
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_initialized()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup_multiplexer()
