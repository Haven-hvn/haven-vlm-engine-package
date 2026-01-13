import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.response import BaseHTTPResponse
import logging
import random
import time
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
from .base_vlm_client import BaseVLMClient

class RetryWithJitter(Retry):
    def __init__(self, *args: Any, jitter_factor: float = 0.25, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.jitter_factor: float = jitter_factor
        if not (0 <= self.jitter_factor <= 1):
            logging.getLogger("logger").warning(
                f"RetryWithJitter initialized with jitter_factor={self.jitter_factor}, which is outside the typical [0, 1] range."
            )

    def sleep(self, response: BaseHTTPResponse | None = None) -> None:
        retry_after: Optional[float] = self.get_retry_after(response=self._last_response)
        if retry_after:
            time.sleep(retry_after)
            return

        backoff_value: float = self.get_backoff_time()
        jitter: float = random.uniform(0, backoff_value * self.jitter_factor)
        sleep_duration: float = backoff_value + jitter

        time.sleep(max(0, sleep_duration))

class OpenAICompatibleVLMClient(BaseVLMClient):
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        super().__init__(config)
        self.api_base_url: str = str(config["api_base_url"]).rstrip('/')
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

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.8,
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
                
                # Log warning if response is empty (model generated no content)
                if not raw_reply or not raw_reply.strip():
                    finish_reason: Optional[str] = response_data.get("choices", [{}])[0].get("finish_reason")
                    usage: Dict[str, Any] = response_data.get("usage", {})
                    completion_tokens: int = usage.get("completion_tokens", 0)
                    self.logger.warning(
                        f"Received empty response from API. "
                        f"Finish reason: {finish_reason}, "
                        f"Completion tokens: {completion_tokens}. "
                        f"This may indicate content filtering, model refusal, or generation issues."
                    )
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
