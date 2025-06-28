import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import base64
from io import BytesIO
from PIL import Image
import logging
import os
import random
import time
from typing import Dict, Any, Optional, List, Tuple, TextIO

class RetryWithJitter(Retry):
    def __init__(self, *args: Any, jitter_factor: float = 0.25, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.jitter_factor: float = jitter_factor
        if not (0 <= self.jitter_factor <= 1):
            logging.getLogger("logger").warning(
                f"RetryWithJitter initialized with jitter_factor={self.jitter_factor}, which is outside the typical [0, 1] range."
            )

    def sleep(self, backoff_value: float) -> None:
        retry_after: Optional[float] = self.get_retry_after(response=self._last_response)
        if retry_after:
            time.sleep(retry_after)
            return

        jitter: float = random.uniform(0, backoff_value * self.jitter_factor)
        sleep_duration: float = backoff_value + jitter
        
        time.sleep(max(0, sleep_duration))

class OpenAICompatibleVLMClient:
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        self.api_base_url: str = str(config["api_base_url"]).rstrip('/')
        self.model_id: str = str(config["model_id"])
        self.max_new_tokens: int = int(config.get("max_new_tokens", 128))
        self.request_timeout: int = int(config.get("request_timeout", 70))
        self.vlm_detected_tag_confidence: float = float(config.get("vlm_detected_tag_confidence", 0.99))
        
        self.tag_list: List[str] = config.get("tag_list")
        if not self.tag_list:
            raise ValueError("Configuration must provide a 'tag_list'.")

        self.logger: logging.Logger = logging.getLogger("logger")
        self.logger.debug(f"VLM Client initialized with {len(self.tag_list)} tags: {self.tag_list[:5]}...")  # Show first 5 tags

        retry_attempts: int = int(config.get("retry_attempts", 3))
        retry_backoff_factor: float = float(config.get("retry_backoff_factor", 0.5))
        retry_jitter_factor: float = float(config.get("retry_jitter_factor", 0.25))
        status_forcelist: Tuple[int, ...] = (500, 502, 503, 504)

        retry_strategy: RetryWithJitter = RetryWithJitter(
            total=retry_attempts,
            backoff_factor=retry_backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=["POST"],
            respect_retry_after_header=True,
            jitter_factor=retry_jitter_factor
        )
        adapter: HTTPAdapter = HTTPAdapter(max_retries=retry_strategy)
        self.session: requests.Session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.logger.info(
            f"Initializing OpenAICompatibleVLMClient for model {self.model_id} "
            f"with {len(self.tag_list)} tags, targeting API: {self.api_base_url}. "
            f"Retry: {retry_attempts} attempts, backoff {retry_backoff_factor}s, jitter factor {retry_jitter_factor}."
        )
        self.logger.info(f"OpenAI VLM client initialized successfully")

    def _convert_image_to_base64_data_url(self, frame: Image.Image, format: str = "JPEG") -> str:
        buffered: BytesIO = BytesIO()
        frame.save(buffered, format=format)
        img_str: str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/{format.lower()};base64,{img_str}"

    async def analyze_frame(self, frame: Optional[Image.Image]) -> Dict[str, float]:
        tag: str
        if not frame:
            self.logger.warning("Analyze_frame called with no frame.")
            return {tag: 0.0 for tag in self.tag_list}

        try:
            image_data_url: str = self._convert_image_to_base64_data_url(frame)
        except Exception as e_convert:
            self.logger.error(f"Failed to convert image to base64: {e_convert}", exc_info=True)
            return {tag: 0.0 for tag in self.tag_list}

        tags_str: str = ", ".join(self.tag_list)
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {
                        "type": "text",
                        "text": (
                            f"What is happening in this scene?"
                        ),
                    },
                ],
            }
        ]

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
            "stream": False,
        }

        api_url: str = f"{self.api_base_url}/v1/chat/completions"
        raw_reply: str = ""
        try:
            response: requests.Response = self.session.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            
            response_data: Dict[str, Any] = response.json()
            if response_data.get("choices") and response_data["choices"][0].get("message"):
                raw_reply = response_data["choices"][0]["message"].get("content", "")
            else:
                self.logger.error(f"Unexpected response structure from API: {response_data}")
                return {tag: 0.0 for tag in self.tag_list}

        except requests.exceptions.RequestException as e_req:
            self.logger.error(f"API request to {api_url} failed: {e_req}", exc_info=True)
            return {tag: 0.0 for tag in self.tag_list}
        except Exception as e_general:
            self.logger.error(f"An unexpected error occurred during API call or response processing: {e_general}", exc_info=True)
            return {tag: 0.0 for tag in self.tag_list}

        return self._parse_simple_default(raw_reply)

    def _parse_simple_default(self, reply: str) -> Dict[str, float]:
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
