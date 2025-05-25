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

# Custom Retry class with jitter
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
        
        tag_list_path: Optional[str] = config.get("tag_list_path") # User provides this path
        self.tag_list: List[str] = [] # Initialize to empty list

        if tag_list_path:
            if os.path.exists(tag_list_path):
                with open(tag_list_path, 'r', encoding='utf-8') as f:
                    f_typed: TextIO = f 
                    self.tag_list = [line.strip() for line in f_typed if line.strip()]
            else:
                # If path provided but not found, it's an issue.
                raise FileNotFoundError(f"Tag list file not found at specified path: {tag_list_path}")
        elif "tag_list" in config and isinstance(config["tag_list"], list):
             self.tag_list = config["tag_list"]
        else:
            # If neither path nor list is provided.
            raise ValueError("Configuration must provide 'tag_list_path' (and file must exist) or a 'tag_list'.")
        
        if not self.tag_list: # Check after attempting to load
            raise ValueError("Tag list is empty. Ensure 'tag_list_path' is correct or 'tag_list' is populated.")

        self.logger: logging.Logger = logging.getLogger("logger")

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
        # Removed "OpenAI VLM client initialized successfully" as it's redundant with the above.

    def _convert_image_to_base64_data_url(self, frame: Image.Image, format: str = "JPEG") -> str:
        buffered: BytesIO = BytesIO()
        frame.save(buffered, format=format.upper()) # Ensure format is uppercase for PIL
        img_str: str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/{format.lower()};base64,{img_str}"

    def analyze_frame(self, frame: Optional[Image.Image]) -> Dict[str, float]:
        tag_key: str # for dict comprehension key
        if not frame:
            self.logger.warning("Analyze_frame called with no frame.")
            return {tag_key: 0.0 for tag_key in self.tag_list}

        try:
            image_data_url: str = self._convert_image_to_base64_data_url(frame)
        except Exception as e_convert:
            self.logger.error(f"Failed to convert image to base64: {e_convert}", exc_info=True)
            return {tag_key: 0.0 for tag_key in self.tag_list}

        # tags_str: str = ", ".join(self.tag_list) # Not used in current payload
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {
                        "type": "text",
                        "text": "What is happening in this scene?", # Generic prompt
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
        self.logger.debug(f"Sending request to {self.model_id} at {api_url} with image.")
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
            if response_data.get("choices") and \
               isinstance(response_data["choices"], list) and \
               response_data["choices"] and \
               isinstance(response_data["choices"][0], dict) and \
               response_data["choices"][0].get("message") and \
               isinstance(response_data["choices"][0]["message"], dict):
                raw_reply = response_data["choices"][0]["message"].get("content", "")
            else:
                self.logger.error(f"Unexpected response structure from API: {response_data}")
                return {tag_key: 0.0 for tag_key in self.tag_list}

            self.logger.debug(f"Response received from {self.model_id}: {raw_reply}")

        except requests.exceptions.RequestException as e_req:
            self.logger.error(f"API request to {api_url} failed: {e_req}", exc_info=True)
            return {tag_key: 0.0 for tag_key in self.tag_list}
        except Exception as e_general:
            self.logger.error(f"An unexpected error occurred during API call or response processing: {e_general}", exc_info=True)
            return {tag_key: 0.0 for tag_key in self.tag_list}

        return self._parse_simple_default(raw_reply)

    def _parse_simple_default(self, reply: str) -> Dict[str, float]:
        found: Dict[str, float] = {tag: 0.0 for tag in self.tag_list}
        parsed_vlm_tags: List[str] = [tag.strip().lower() for tag in reply.split(',') if tag.strip()]

        for tag_config_original_case in self.tag_list:
            if tag_config_original_case.lower() in parsed_vlm_tags:
                found[tag_config_original_case] = self.vlm_detected_tag_confidence
        
        return found

    def _parse_score_per_line(self, reply: str) -> Dict[str, float]:
        # Placeholder for a different parsing strategy if needed
        # This method is not currently used by analyze_frame.
        self.logger.warning("_parse_score_per_line is not implemented.")
        return {tag: 0.0 for tag in self.tag_list}
