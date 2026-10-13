import asyncio
import logging
import random
import warnings
import time
import json
from typing import Dict, Any, Optional, List
from PIL import Image
import httpx2
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
    - Concurrency control has moved from network layer (httpx2 limits) to application layer (native slot reservation)
    - Intelligent overflow routing, immediate self-healing rebalancing, and weighted distribution preservation under load
    - The Global Semaphore (self.semaphore) remains the primary guard against system-wide overload
    - Per-endpoint concurrency limits are now configured via max_concurrent parameter
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Performance optimization settings
        self.max_concurrent_requests: int = int(
            config.get("max_concurrent_requests", 20)
        )

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
                    stacklevel=2,
                )

        self.logger.debug(
            f"MultiplexerVLMClient initialized with {len(self.tag_list)} tags: {self.tag_list[:5]}..."
        )

        # Extract multiplexer endpoints configuration
        self.multiplexer_endpoints: List[Dict[str, Any]] = config.get(
            "multiplexer_endpoints", []
        )
        if not self.multiplexer_endpoints:
            raise ValueError(
                "Configuration must provide 'multiplexer_endpoints' for multiplexer mode."
            )

        # Extract optional System One (Jev /v1/systemone) endpoints.
        # Jev endpoints only speak /v1/systemone, so they are routed through a
        # dedicated multiplexer pool instead of joining the chat/VLM pool above.
        self.systemone_endpoints: List[Dict[str, Any]] = config.get(
            "systemone_endpoints", []
        )

        # Initialize concurrency control
        self.semaphore = asyncio.Semaphore(
            self.max_concurrent_requests
        )  # Global admission control
        self.multiplexer: Optional[Multiplexer] = None
        self._initialized = False

        # System One (Jev /v1/systemone) pool - dedicated multiplexer, lazily
        # initialized so chat-only configurations never pay for it
        self.systemone_multiplexer: Optional[Multiplexer] = None
        self._systemone_initialized = False

        # Debug instrumentation for hypothesis B (connection pool exhaustion)
        self.client_requests_count = 0
        self.concurrency_active_count = 0

        self.logger.info(
            f"Initializing high-performance MultiplexerVLMClient for model {self.model_id} "
            f"with {len(self.tag_list)} tags, {len(self.multiplexer_endpoints)} endpoints, "
            f"global_max_concurrent: {self.max_concurrent_requests}"
        )

        if self.systemone_endpoints:
            self.logger.info(
                f"System One (Jev /v1/systemone) pool configured with {len(self.systemone_endpoints)} endpoints "
                f"(kept separate from the chat/VLM endpoint pool)"
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

        self.logger.info(
            "Initializing high-performance multiplexer with application-layer concurrency control..."
        )

        # Create multiplexer instance
        self.multiplexer = Multiplexer()
        await self.multiplexer.__aenter__()

        # Add endpoints to multiplexer with per-endpoint concurrency control
        # Move httpx2 limits calculation inside endpoint loop for per-node optimization
        for i, endpoint_config in enumerate(self.multiplexer_endpoints):
            try:
                self._register_endpoint(self.multiplexer, endpoint_config, i)
            except Exception as e:
                self.logger.error(f"Failed to add endpoint {endpoint_config}: {e}")
                raise

        self._initialized = True
        self.logger.info(
            f"Multiplexer initialization completed with {len(self.multiplexer_endpoints)} endpoints and application-layer concurrency control"
        )

    def _register_endpoint(
        self, multiplexer: Multiplexer, endpoint_config: Dict[str, Any], index: int
    ) -> None:
        """Register a single endpoint with a multiplexer pool.

        Shared by both pools. Each pool is homogeneous by construction: the chat
        pool (multiplexer_endpoints) only ever serves chat.completions.create and
        the System One pool (systemone_endpoints) only ever serves systemone.create,
        so the two endpoint types are never mixed within one pool.
        """
        # Calculate per-endpoint connection limits based on that specific node's capacity
        # Use per-endpoint max_concurrent for sizing, with reasonable default for pipelining
        per_node_max_concurrent = endpoint_config.get(
            "max_concurrent", 3
        )  # Default to 3 for pipelining
        limit_buffer = 5  # Small safety buffer since per-node limits
        per_node_limits = httpx2.Limits(
            max_keepalive_connections=per_node_max_concurrent + limit_buffer,
            max_connections=(per_node_max_concurrent * 2) + limit_buffer,
            keepalive_expiry=30.0,
        )

        # Create HTTP client with per-node connection limits
        http_client: httpx2.AsyncClient = httpx2.AsyncClient(
            timeout=httpx2.Timeout(self.request_timeout),
            limits=per_node_limits,  # Per-node limits instead of global hardcoded values
        )

        # Create AsyncOpenAI client with optimized HTTP client
        # max_retries=0 lets multiplexer-llm handle failover immediately
        client = AsyncOpenAI(
            api_key=endpoint_config.get("api_key", "dummy_api_key"),
            base_url=endpoint_config["base_url"],
            http_client=http_client,
            max_retries=0,  # Let multiplexer-llm handle failover immediately
            timeout=self.request_timeout,
        )

        weight = endpoint_config.get("weight", 1)
        name = endpoint_config.get("name", f"endpoint-{index}")
        is_fallback = endpoint_config.get("is_fallback", False)
        # NEW: Extract max_concurrent from endpoint config (defaults to None for unlimited)
        max_concurrent = endpoint_config.get("max_concurrent")

        if is_fallback:
            multiplexer.add_fallback_model(
                client, weight, name, max_concurrent=max_concurrent
            )
            self.logger.info(
                f"Added fallback endpoint: {name} (weight: {weight}, max_concurrent: {max_concurrent or 'unlimited'})"
            )
        else:
            multiplexer.add_model(
                client, weight, name, max_concurrent=max_concurrent
            )
            self.logger.info(
                f"Added primary endpoint: {name} (weight: {weight}, max_concurrent: {max_concurrent or 'unlimited'})"
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

    async def _ensure_systemone_initialized(self):
        """Ensure the System One multiplexer is initialized. Called before each query.

        Idempotent - safe to call multiple times; only the first call initializes.
        The pool is built lazily so clients without systemone_endpoints never
        pay for the second multiplexer.
        """
        if not self._systemone_initialized:
            await self._initialize_systemone_multiplexer()

    async def _initialize_systemone_multiplexer(self):
        """Initialize the dedicated System One (Jev /v1/systemone) endpoint pool."""
        if self._systemone_initialized:
            return

        self.logger.info(
            f"Initializing System One multiplexer with {len(self.systemone_endpoints)} endpoints..."
        )

        # Create a SEPARATE multiplexer instance for the System One pool.
        # Jev endpoints only speak /v1/systemone while VLM endpoints only speak
        # /chat/completions, so each pool gets its own multiplexer and the two
        # endpoint types are never mixed within one pool.
        self.systemone_multiplexer = Multiplexer()
        await self.systemone_multiplexer.__aenter__()

        for i, endpoint_config in enumerate(self.systemone_endpoints):
            try:
                self._register_endpoint(self.systemone_multiplexer, endpoint_config, i)
            except Exception as e:
                self.logger.error(f"Failed to add System One endpoint {endpoint_config}: {e}")
                raise

        self._systemone_initialized = True
        self.logger.info(
            "System One multiplexer initialization completed with application-layer concurrency control"
        )

    async def _cleanup_systemone_multiplexer(self):
        """Cleanup System One multiplexer resources."""
        if self.systemone_multiplexer and self._systemone_initialized:
            try:
                await self.systemone_multiplexer.__aexit__(None, None, None)
                self.logger.info("System One multiplexer cleanup completed")
            except Exception as e:
                self.logger.error(f"Error during System One multiplexer cleanup: {e}")
            finally:
                self.systemone_multiplexer = None
                self._systemone_initialized = False

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
                self.logger.error(
                    f"Failed to convert image to base64: {e_convert}", exc_info=True
                )
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

            # Implement exponential backoff with jitter for transient errors
            # Note: multiplexer-llm handles per-endpoint failover internally (tries all endpoints)
            # These errors are only raised when ALL endpoints (primary + fallback) are exhausted
            # We backoff here to give the cluster time to recover before retrying
            max_attempts = 500  # Many more attempts for resilience
            base_delay = 30  # Start at 30 seconds
            max_delay = 120  # 2 minute max per attempt
            backoff_factor = 1.2

            for attempt in range(max_attempts):
                try:
                    # Use multiplexer for the request with proper exception handling
                    completion = await self.multiplexer.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        max_tokens=self.max_new_tokens,
                        temperature=0.8,
                        timeout=self.request_timeout,
                    )

                    if completion.choices and completion.choices[0].message:
                        raw_reply = completion.choices[0].message.content or ""

                        # Log warning if response is empty (model generated no content)
                        if not raw_reply or not raw_reply.strip():
                            finish_reason: Optional[str] = getattr(
                                completion.choices[0], "finish_reason", None
                            )
                            usage: Optional[Any] = getattr(completion, "usage", None)
                            completion_tokens: int = 0
                            if usage:
                                completion_tokens = getattr(
                                    usage, "completion_tokens", 0
                                )
                            self.logger.warning(
                                f"Received empty response from multiplexer. "
                                f"Finish reason: {finish_reason}, "
                                f"Completion tokens: {completion_tokens}. "
                                f"This may indicate content filtering, model refusal, or generation issues."
                            )
                        else:
                            self.logger.debug(
                                f"Received response from multiplexer: {raw_reply[:100]}..."
                            )

                        self.concurrency_active_count -= 1

                        return self._parse_simple_default(raw_reply)
                    else:
                        self.logger.error(
                            f"Unexpected response structure from multiplexer: {completion}"
                        )
                        self.concurrency_active_count -= 1
                        return {tag: 0.0 for tag in self.tag_list}

                except (
                    RateLimitError,
                    ModelSelectionError,
                    ServiceUnavailableError,
                ) as e:
                    if attempt == max_attempts - 1:
                        error_type = type(e).__name__
                        self.logger.error(
                            f"{error_type} persisted after {max_attempts} attempts "
                            f"({max_attempts * max_delay / 60:.0f} minutes total). Giving up: {e.message}"
                        )
                        self.concurrency_active_count -= 1
                        raise

                    retry_after = getattr(e, "retry_after", None)
                    if retry_after:
                        wait_time = min(retry_after, max_delay)
                    else:
                        wait_time = min(
                            base_delay * (backoff_factor**attempt), max_delay
                        )

                    wait_time *= 1 + random.random() * 0.5

                    error_type = type(e).__name__
                    self.logger.warning(
                        f"{error_type}. Waiting {wait_time / 60:.1f}m (attempt {attempt + 1}/{max_attempts})..."
                    )
                    await asyncio.sleep(wait_time)

                except ModelNotFoundError as e:
                    self.logger.error(f"Model not found: {e.message}")
                    self.concurrency_active_count -= 1
                    return {tag: 0.0 for tag in self.tag_list}

                except AuthenticationError as e:
                    self.logger.error(f"Authentication failed: {e.message}")
                    self.concurrency_active_count -= 1
                    return {tag: 0.0 for tag in self.tag_list}

                except MultiplexerError as e:
                    self.logger.error(f"Multiplexer error: {e.message}")
                    self.concurrency_active_count -= 1
                    return {tag: 0.0 for tag in self.tag_list}

                except Exception as e:
                    self.logger.error(
                        f"Unexpected error during frame analysis: {e}", exc_info=True
                    )
                    self.concurrency_active_count -= 1
                    return {tag: 0.0 for tag in self.tag_list}

    async def ask_systemone(
        self,
        *,
        state: Any,
        questions: Dict[str, Any],
        path: str = "/systemone",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Ask atomic questions about a state via the Jev /v1/systemone contract.

        Routed through the dedicated System One endpoint pool
        (systemone_endpoints) with the same weighted selection, capacity
        reservation, and automatic failover as chat completions. Unlike
        analyze_frame, errors are raised rather than folded into
        zero-confidence tags - the caller owns the fallback behavior for
        decisions.

        Args:
            state: The subject being judged (plain text or structured JSON).
            questions: Map of question id -> question body (noul/choice/score).
                Ids are echoed in the response's answers.
            path: Endpoint path. Defaults to "/systemone" and is merged onto
                each client's base_url (which should include the /v1 prefix).
            **kwargs: Additional request body fields (e.g. choice_head).

        Returns:
            The parsed response body as a dict (model, answers, usage).

        Raises:
            ValueError: No systemone_endpoints configured.
            RateLimitError/ModelSelectionError/ServiceUnavailableError: All
                endpoints exhausted after the full retry budget.
            AuthenticationError/ModelNotFoundError/MultiplexerError: Pool
                misconfigured or endpoint rejected the request.
        """
        if not self.systemone_endpoints:
            raise ValueError(
                "Configuration must provide 'systemone_endpoints' to ask System One questions."
            )

        # Use semaphore for GLOBAL admission control (system chokehold),
        # shared with analyze_frame so the two surfaces cannot jointly
        # overload the host process
        async with self.semaphore:
            self.client_requests_count += 1
            self.concurrency_active_count += 1
            try:
                await self._ensure_systemone_initialized()

                # Same resilience profile as analyze_frame: multiplexer-llm
                # fails over across endpoints internally (429/529 -> 60s
                # disable, 5xx -> 15s); these errors surface only when ALL
                # endpoints (primary + fallback) are exhausted, so we back off
                # to give the cluster time to recover before retrying
                max_attempts = 500  # Many more attempts for resilience
                base_delay = 30  # Start at 30 seconds
                max_delay = 120  # 2 minute max per attempt
                backoff_factor = 1.2

                for attempt in range(max_attempts):
                    try:
                        # Use the dedicated System One multiplexer for the request
                        return await self.systemone_multiplexer.systemone.create(
                            state=state,
                            questions=questions,
                            path=path,
                            **kwargs,
                        )
                    except (
                        RateLimitError,
                        ModelSelectionError,
                        ServiceUnavailableError,
                    ) as e:
                        if attempt == max_attempts - 1:
                            error_type = type(e).__name__
                            self.logger.error(
                                f"{error_type} persisted after {max_attempts} attempts "
                                f"({max_attempts * max_delay / 60:.0f} minutes total). Giving up: {e.message}"
                            )
                            raise

                        retry_after = getattr(e, "retry_after", None)
                        if retry_after:
                            wait_time = min(retry_after, max_delay)
                        else:
                            wait_time = min(
                                base_delay * (backoff_factor**attempt), max_delay
                            )

                        wait_time *= 1 + random.random() * 0.5

                        error_type = type(e).__name__
                        self.logger.warning(
                            f"{error_type} while asking System One questions. Waiting {wait_time / 60:.1f}m (attempt {attempt + 1}/{max_attempts})..."
                        )
                        await asyncio.sleep(wait_time)

                    except ModelNotFoundError as e:
                        # Misconfiguration, not a transient condition - surface it
                        self.logger.error(f"System One model not found: {e.message}")
                        raise

                    except AuthenticationError as e:
                        # Misconfiguration, not a transient condition - surface it
                        self.logger.error(f"System One authentication failed: {e.message}")
                        raise

                    except MultiplexerError as e:
                        # Anything else multiplexer-related (e.g. an invalid
                        # response body) is not worth retrying across the pool
                        self.logger.error(f"Multiplexer error during System One query: {e.message}")
                        raise
            finally:
                self.concurrency_active_count -= 1

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup_multiplexer()
        await self._cleanup_systemone_multiplexer()
