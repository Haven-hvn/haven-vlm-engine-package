import asyncio
import logging
import warnings
import time
import json
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
    
    NEW ARCHITECTURE:
    - Concurrency control has moved from network layer (httpx limits) to application layer (native slot reservation)
    - Intelligent overflow routing, immediate self-healing rebalancing, and weighted distribution preservation under load
    - The Global Semaphore (self.semaphore) remains the primary guard against system-wide overload
    - Per-endpoint concurrency limits are now configured via max_concurrent parameter
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Performance optimization settings
        self.max_concurrent_requests: int = int(config.get("max_concurrent_requests", 20))
        
        # connection_pool_size is deprecated - multiplexer now handles connection management
        # Keeping for backward compatibility but it's no longer used for controlling concurrency
        # Only warn if user explicitly set this parameter (not just using default value)
        if "connection_pool_size" in config:
            connection_pool_size_val = int(config.get("connection_pool_size", 50))
            if connection_pool_size_val != 50:
                warnings.warn(
                    "connection_pool_size parameter is deprecated: The multiplexer now manages connection pools internally. "
                    "This parameter will be removed in a future version.",
                    DeprecationWarning,
                    stacklevel=2
                )
        
        self.logger.debug(f"MultiplexerVLMClient initialized with {len(self.tag_list)} tags: {self.tag_list[:5]}...")
        
        # Extract multiplexer endpoints configuration
        self.multiplexer_endpoints: List[Dict[str, Any]] = config.get("multiplexer_endpoints", [])
        if not self.multiplexer_endpoints:
            raise ValueError("Configuration must provide 'multiplexer_endpoints' for multiplexer mode.")
        
        # Initialize concurrency control
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)  # Global admission control
        self.multiplexer: Optional[Multiplexer] = None
        self._initialized = False
        
        # Debug instrumentation for hypothesis B (connection pool exhaustion)
        self.client_requests_count = 0
        self.concurrency_active_count = 0
        
        self.logger.info(
            f"Initializing high-performance MultiplexerVLMClient for model {self.model_id} "
            f"with {len(self.tag_list)} tags, {len(self.multiplexer_endpoints)} endpoints, "
            f"global_max_concurrent: {self.max_concurrent_requests}"
        )

    async def _ensure_initialized(self):
        """Ensure the multiplexer is initialized. Called before each request.
        
        This method is idempotent - safe to call multiple times. It will only initialize once
        and subsequent calls will be no-ops due to the _initialized flag check.
        """
        if not self._initialized:
            await self._initialize_multiplexer()
    
    async def _initialize_multiplexer(self):
        """Initialize the multiplexer with configured endpoints and application-layer concurrency control."""
        if self._initialized:
            return
            
        self.logger.info("Initializing high-performance multiplexer with application-layer concurrency control...")
        
        # Create multiplexer instance
        self.multiplexer = Multiplexer()
        await self.multiplexer.__aenter__()
        
        # Add endpoints to multiplexer with per-endpoint concurrency control
        # Move httpx limits calculation inside endpoint loop for per-node optimization
        for i, endpoint_config in enumerate(self.multiplexer_endpoints):
            try:
                # Calculate per-endpoint connection limits based on that specific node's capacity
                # Use per-endpoint max_concurrent for sizing, with reasonable default for pipelining
                per_node_max_concurrent = endpoint_config.get("max_concurrent", 3)  # Default to 3 for pipelining
                limit_buffer = 5  # Small safety buffer since per-node limits
                per_node_limits = httpx.Limits(
                    max_keepalive_connections=per_node_max_concurrent + limit_buffer,
                    max_connections=(per_node_max_concurrent * 2) + limit_buffer,
                    keepalive_expiry=30.0
                )
                
                # Create HTTP client with per-node connection limits
                http_client: httpx.AsyncClient = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.request_timeout),
                    limits=per_node_limits  # Per-node limits instead of global hardcoded values
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
                # NEW: Extract max_concurrent from endpoint config (defaults to None for unlimited)
                max_concurrent = endpoint_config.get("max_concurrent")
                
                if is_fallback:
                    self.multiplexer.add_fallback_model(client, weight, name, max_concurrent=max_concurrent)
                    self.logger.info(f"Added fallback endpoint: {name} (weight: {weight}, max_concurrent: {max_concurrent or 'unlimited'})")
                else:
                    self.multiplexer.add_model(client, weight, name, max_concurrent=max_concurrent)
                    self.logger.info(f"Added primary endpoint: {name} (weight: {weight}, max_concurrent: {max_concurrent or 'unlimited'})")
                    
            except Exception as e:
                self.logger.error(f"Failed to add endpoint {endpoint_config}: {e}")
                raise
        
        self._initialized = True
        self.logger.info(
            f"Multiplexer initialization completed with {len(self.multiplexer_endpoints)} endpoints and application-layer concurrency control"
        )
    
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
        
        Features:
        - Global admission control via semaphore (system chokehold)
        - Application-layer concurrency control via Multiplexer max_concurrent
        - Intelligent overflow routing when endpoints hit capacity
        - Immediate self-healing rebalancing
        - Preserved weighted distribution under normal load
        """
        if not frame:
            self.logger.warning("analyze_frame called with no frame.")
            return {tag: 0.0 for tag in self.tag_list}
        
        # Debug instrumentation for hypothesis B (HTTP request lifecycle)
        frame_start_time = time.time()
        request_id = f"{int(frame_start_time * 1000000)}"  # microsecond timestamp

        # Use semaphore for GLOBAL admission control (system chokehold)
        async with self.semaphore:
            # Ensure multiplexer is initialized
            await self._ensure_initialized()
            
            try:
                self.client_requests_count += 1
                self.concurrency_active_count += 1
                image_data_url: str = self._convert_image_to_base64_data_url(frame)
            except Exception as e_convert:
                self.logger.error(f"Failed to convert image to base64: {e_convert}", exc_info=True)
                self.concurrency_active_count -= 1
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
                    temperature=0.8,
                    timeout=self.request_timeout
                )
                
                if completion.choices and completion.choices[0].message:
                    raw_reply = completion.choices[0].message.content or ""
                    
                    # Log warning if response is empty (model generated no content)
                    if not raw_reply or not raw_reply.strip():
                        finish_reason: Optional[str] = getattr(completion.choices[0], "finish_reason", None)
                        usage: Optional[Any] = getattr(completion, "usage", None)
                        completion_tokens: int = 0
                        if usage:
                            completion_tokens = getattr(usage, "completion_tokens", 0)
                        self.logger.warning(
                            f"Received empty response from multiplexer. "
                            f"Finish reason: {finish_reason}, "
                            f"Completion tokens: {completion_tokens}. "
                            f"This may indicate content filtering, model refusal, or generation issues."
                        )
                    else:
                        self.logger.debug(f"Received response from multiplexer: {raw_reply[:100]}...")
                    
                    self.concurrency_active_count -= 1

                    return self._parse_simple_default(raw_reply)
                else:
                    self.logger.error(f"Unexpected response structure from multiplexer: {completion}")
                    self.concurrency_active_count -= 1
                    return {tag: 0.0 for tag in self.tag_list}
                    
            except ModelNotFoundError as e:
                self.logger.error(f"Model not found at endpoint {e.endpoint}: {e.message}")
                self.concurrency_active_count -= 1
                return {tag: 0.0 for tag in self.tag_list}
                
            except AuthenticationError as e:
                self.logger.error(f"Authentication failed at endpoint {e.endpoint}: {e.message}")
                self.concurrency_active_count -= 1
                return {tag: 0.0 for tag in self.tag_list}
                
            except RateLimitError as e:
                self.logger.warning(f"Rate limit hit at endpoint {e.endpoint}, retry after {e.retry_after}s: {e.message}")
                self.concurrency_active_count -= 1
                return {tag: 0.0 for tag in self.tag_list}
                
            except ServiceUnavailableError as e:
                self.logger.error(f"Service unavailable at endpoint {e.endpoint}: {e.message}")
                self.concurrency_active_count -= 1
                return {tag: 0.0 for tag in self.tag_list}
                
            except ModelSelectionError as e:
                self.logger.error(f"No models available for selection: {e.message}")
                self.concurrency_active_count -= 1
                return {tag: 0.0 for tag in self.tag_list}
                
            except MultiplexerError as e:
                self.logger.error(f"Multiplexer error: {e.message}")
                self.concurrency_active_count -= 1
                return {tag: 0.0 for tag in self.tag_list}
                
            except Exception as e:
                self.logger.error(f"Unexpected error during frame analysis: {e}", exc_info=True)
                self.concurrency_active_count -= 1

                return {tag: 0.0 for tag in self.tag_list}
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_initialized()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup_multiplexer()
