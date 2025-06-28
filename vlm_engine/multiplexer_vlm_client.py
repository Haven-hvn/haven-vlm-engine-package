import asyncio
import base64
from io import BytesIO
from PIL import Image
import logging
from typing import Dict, Any, Optional, List
from multiplexer_llm import Multiplexer
from openai import AsyncOpenAI


class MultiplexerVLMClient:
    """
    VLM client that uses multiplexer-llm for load balancing across multiple OpenAI-compatible endpoints.
    Maintains the same interface as OpenAICompatibleVLMClient for seamless integration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.model_id: str = str(config["model_id"])
        self.max_new_tokens: int = int(config.get("max_new_tokens", 128))
        self.request_timeout: int = int(config.get("request_timeout", 70))
        self.vlm_detected_tag_confidence: float = float(config.get("vlm_detected_tag_confidence", 0.99))
        
        self.tag_list: List[str] = config.get("tag_list")
        if not self.tag_list:
            raise ValueError("Configuration must provide a 'tag_list'.")
        
        self.logger: logging.Logger = logging.getLogger("logger")
        self.logger.debug(f"MultiplexerVLMClient initialized with {len(self.tag_list)} tags: {self.tag_list[:5]}...")
        
        # Extract multiplexer endpoints configuration
        self.multiplexer_endpoints: List[Dict[str, Any]] = config.get("multiplexer_endpoints", [])
        if not self.multiplexer_endpoints:
            raise ValueError("Configuration must provide 'multiplexer_endpoints' for multiplexer mode.")
        
        self.multiplexer: Optional[Multiplexer] = None
        self._initialized = False
        
        self.logger.info(
            f"Initializing MultiplexerVLMClient for model {self.model_id} "
            f"with {len(self.tag_list)} tags and {len(self.multiplexer_endpoints)} endpoints."
        )
    
    async def _ensure_initialized(self):
        """Ensure the multiplexer is initialized. Called before each request."""
        if not self._initialized:
            await self._initialize_multiplexer()
    
    async def _initialize_multiplexer(self):
        """Initialize the multiplexer with configured endpoints."""
        if self._initialized:
            return
            
        self.logger.info("Initializing multiplexer with endpoints...")
        
        # Create multiplexer instance
        self.multiplexer = Multiplexer()
        await self.multiplexer.__aenter__()
        
        # Add endpoints to multiplexer
        for i, endpoint_config in enumerate(self.multiplexer_endpoints):
            try:
                # Create AsyncOpenAI client for this endpoint
                client = AsyncOpenAI(
                    api_key=endpoint_config.get("api_key", "dummy_api_key"),
                    base_url=endpoint_config["base_url"],
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
        self.logger.info("Multiplexer initialization completed successfully")
    
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
    
    def _convert_image_to_base64_data_url(self, frame: Image.Image, format: str = "JPEG") -> str:
        """Convert PIL Image to base64 data URL."""
        buffered: BytesIO = BytesIO()
        frame.save(buffered, format=format)
        img_str: str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/{format.lower()};base64,{img_str}"
    
    async def analyze_frame(self, frame: Optional[Image.Image]) -> Dict[str, float]:
        """
        Analyze a frame using the multiplexer for load balancing across endpoints.
        Maintains the same interface as OpenAICompatibleVLMClient.
        """
        if not frame:
            self.logger.warning("analyze_frame called with no frame.")
            return {tag: 0.0 for tag in self.tag_list}
        
        # Ensure multiplexer is initialized
        await self._ensure_initialized()
        
        try:
            image_data_url: str = self._convert_image_to_base64_data_url(frame)
        except Exception as e_convert:
            self.logger.error(f"Failed to convert image to base64: {e_convert}", exc_info=True)
            return {tag: 0.0 for tag in self.tag_list}
        
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {
                        "type": "text",
                        "text": "What is happening in this scene?",
                    },
                ],
            }
        ]
        
        try:
            # Use multiplexer for the request
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
            else:
                self.logger.error(f"Unexpected response structure from multiplexer: {completion}")
                return {tag: 0.0 for tag in self.tag_list}
                
        except Exception as e:
            self.logger.error(f"Multiplexer request failed: {e}", exc_info=True)
            return {tag: 0.0 for tag in self.tag_list}
        
        return self._parse_simple_default(raw_reply)
    
    def _parse_simple_default(self, reply: str) -> Dict[str, float]:
        """Parse VLM response to extract detected tags."""
        found: Dict[str, float] = {tag: 0.0 for tag in self.tag_list}
        
        # First strip the entire reply to remove leading/trailing whitespace
        reply = reply.strip()
        
        # Split by comma and strip each tag
        parsed_vlm_tags: List[str] = [tag.strip().lower() for tag in reply.split(',') if tag.strip()]
        
        # Log the parsed tags for debugging
        self.logger.debug(f"VLM raw reply: '{reply}'")
        self.logger.debug(f"Parsed VLM tags (lowercase): {parsed_vlm_tags}")
        
        for tag_config_original_case in self.tag_list:
            if tag_config_original_case.lower() in parsed_vlm_tags:
                found[tag_config_original_case] = self.vlm_detected_tag_confidence
                self.logger.debug(f"Matched tag: '{tag_config_original_case}' with confidence {self.vlm_detected_tag_confidence}")
        
        return found
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_initialized()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup_multiplexer()
