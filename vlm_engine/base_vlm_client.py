import base64
from io import BytesIO
from PIL import Image
import logging
import re
from typing import Dict, Any, Optional, List


class BaseVLMClient:
    """
    Base class for VLM clients containing shared parsing, prompt generation, and utility methods.
    Eliminates code duplication between OpenAICompatibleVLMClient and MultiplexerVLMClient.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize common configuration shared by all VLM clients."""
        self.model_id: str = str(config["model_id"])
        # Default to 131072 (128K tokens) - GLM max context length, good middle ground for modern models
        # One-shot prompts have low risk, no context accumulation across frames
        self.max_new_tokens: int = int(config.get("max_new_tokens", 131072))
        self.request_timeout: int = int(config.get("request_timeout", 600))
        self.vlm_detected_tag_confidence: float = float(config.get("vlm_detected_tag_confidence", 0.99))
        
        self.tag_list: List[str] = config.get("tag_list")
        if not self.tag_list:
            raise ValueError("Configuration must provide a 'tag_list'.")

        self.logger: logging.Logger = logging.getLogger("logger")

        # Parse optional special_tokens config (model-specific)
        special_tokens_config: Optional[Dict[str, str]] = config.get("special_tokens")
        self.special_tokens: Optional[Dict[str, str]] = None
        if special_tokens_config and isinstance(special_tokens_config, dict):
            begin_token: Optional[str] = special_tokens_config.get("begin")
            end_token: Optional[str] = special_tokens_config.get("end")
            if begin_token and end_token:
                self.special_tokens = {"begin": str(begin_token), "end": str(end_token)}
                self.logger.info(f"Special tokens configured: begin='{begin_token}', end='{end_token}'")

    def _convert_image_to_base64_data_url(self, frame: Image.Image, format: str = "JPEG") -> str:
        """Convert PIL Image to base64 data URL."""
        buffered: BytesIO = BytesIO()
        frame.save(buffered, format=format)
        img_str: str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/{format.lower()};base64,{img_str}"

    def _normalize_text(self, text: str) -> str:
        """Normalize text for robust matching: lowercase, remove punctuation, normalize whitespace."""
        if not text:
            return ""
        
        # Convert to lowercase
        normalized: str = text.lower()
        # Replace punctuation and special characters with spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        # Collapse multiple whitespace to single space
        normalized = re.sub(r'\s+', ' ', normalized)
        # Strip leading/trailing whitespace
        return normalized.strip()

    def _format_tag_list_for_prompt(self) -> str:
        """Format tag list for prompt, using category grouping for large lists."""
        tag_count: int = len(self.tag_list)
        
        # For small lists, return simple comma-separated format
        if tag_count < 30:
            return ", ".join(self.tag_list)
        
        # For large lists, try to group by category if category info available
        # TODO: Could add category metadata to tag_list in future
        # For now, fall back to comma-separated (models can handle it)
        # Alternative: Group by first letter or semantic similarity
        return ", ".join(self.tag_list)

    def _extract_from_special_tokens(self, reply: str) -> str:
        """
        Extract content from special tokens.
        First tries configured special_tokens if provided, then falls back to detecting common patterns.
        This makes the parser dynamic to handle different models even without explicit configuration.
        """
        # Try configured special tokens first (model-specific)
        if self.special_tokens:
            begin_token: Optional[str] = self.special_tokens.get("begin")
            end_token: Optional[str] = self.special_tokens.get("end")
            
            if begin_token and end_token:
                # Extract all content between begin and end tokens
                pattern: str = re.escape(begin_token) + r'(.*?)' + re.escape(end_token)
                matches: List[str] = re.findall(pattern, reply, re.DOTALL)
                if matches:
                    # Join all extracted content with spaces
                    return ' '.join(matches)
        
        # Fallback: Detect common special token patterns even if not configured
        # This makes the parser work with models like GLM that use special tokens by default
        common_patterns: List[tuple[str, str]] = [
            (r'<\|begin_of_box\|>', r'<\|end_of_box\|>'),  # GLM pattern
            (r'<\|beginoftext\|>', r'<\|endoftext\|>'),     # Common pattern
            (r'\[START\]', r'\[END\]'),                     # Simple pattern
        ]
        
        for begin_pattern, end_pattern in common_patterns:
            pattern: str = begin_pattern + r'(.*?)' + end_pattern
            matches: List[str] = re.findall(pattern, reply, re.DOTALL)
            if matches:
                # Found a match, return extracted content
                return ' '.join(matches)
        
        # No special tokens found
        return reply

    def _build_prompt_text(self) -> str:
        """Build the enhanced two-step prompt text for VLM analysis."""
        # Format tag list for prompt
        tag_list_str: str = self._format_tag_list_for_prompt()
        
        # Build improved prompt with clearer instructions and structure
        output_format: str = "[description] | [single_tag] or [description] | none"
        if self.special_tokens:
            output_format = f"[description] | [single_tag] or use special tokens {self.special_tokens['begin']}tag{self.special_tokens['end']} for the tag"
        
        prompt_text: str = (
            f"Analyze this image and identify what is present. Focus ONLY on what you actually see, not what might be there.\n\n"
            f"Step 1: Briefly describe the key visual elements (MAX 15 WORDS): actions, objects, scenes, and relationships you observe.\n"
            f"Do not over-analyze or interpret beyond visual evidence.\n\n"
            f"Step 2: From this tag list, select the SINGLE BEST tag that EXACTLY matches what you described: {tag_list_str}\n"
            f"Important:\n"
            f"- Select ONLY ONE tag - the most appropriate tag for this frame\n"
            f"- Prefer MORE SPECIFIC tags over general ones (e.g., 'dribbling' over 'basketball', 'jump shot' over 'shooting')\n"
            f"- If tags are mutually exclusive (e.g., 'defense' vs 'offense'), select the one that is actually happening\n"
            f"- Choose the tag that is MOST REPRESENTATIVE of what's clearly visible in this frame\n"
            f"- Only include a tag if it CLEARLY matches the tag definition\n"
            f"- Tags must match EXACTLY - do not apply tags to similar but different content\n"
            f"- Do not check every tag - focus only on what's actually present\n"
            f"- If no tags match, use \"none\"\n\n"
            f"Output format: {output_format}\n\n"
            f"Requirements:\n"
            f"- Keep the description concise (MAX 15 WORDS, before the | delimiter)\n"
            f"- Select exactly one tag that appears in the list (after the | delimiter)\n"
            f"- If no tags match, use: [description] | none"
        )
        return prompt_text

    def _parse_simple_default(self, reply: str) -> Dict[str, float]:
        """
        Parse VLM response with two-tier approach:
        - Primary path: Simple comma-split parsing (exact format as demanded)
        - Fallback path: Normalization-based parsing (only if primary finds nothing)
        """
        # Initialize all configured tags (from self.tag_list, which preserves original casing) with 0.0 confidence
        found: Dict[str, float] = {tag: 0.0 for tag in self.tag_list}
        
        # Step 1: Handle empty responses and "none" variations (edge case handling)
        if not reply or not reply.strip():
            return found
        
        normalized_reply: str = reply.lower().strip()
        if normalized_reply in ['none', 'nothing', 'no tags', 'n/a', 'na']:
            return found
        
        # PRIMARY PATH: Simple split-based parsing (exact format as demanded)
        # Step 2: Extract content from special tokens FIRST (if configured)
        # This handles cases where special tokens contain the structured format
        content: str = self._extract_from_special_tokens(reply)
        
        # Step 3: Extract tags section (after | delimiter if present, or use entire content)
        if '|' in content:
            # Extract content after | delimiter (tags section)
            parts: List[str] = content.split('|', 1)
            if len(parts) > 1:
                content = parts[1].strip()  # Get tags part after delimiter
        
        # Step 3.5: Handle "none" after delimiter (explicit "no tags" case)
        if content:
            normalized_content_check: str = content.lower().strip()
            if normalized_content_check in ['none', 'nothing', 'no tags', 'n/a', 'na']:
                return found
        
        # Step 4: Simple comma-split parsing (primary path)
        tags_found_primary: bool = False
        if content:
            # Split by comma and clean up
            parsed_tags: List[str] = [tag.strip().lower() for tag in content.split(',') if tag.strip()]
            
            # Match exactly against tag_list (case-insensitive)
            for tag in self.tag_list:
                if tag.lower() in parsed_tags:
                    found[tag] = self.vlm_detected_tag_confidence
                    tags_found_primary = True
        
        # FALLBACK PATH: Normalization-based parsing (only if primary found nothing)
        if not tags_found_primary:
            # Normalize the content (handles punctuation, whitespace, case, unicode)
            normalized_content: str = self._normalize_text(content if content else reply)
            
            if normalized_content:
                # For each configured tag, check if it appears as whole word/phrase
                for tag in self.tag_list:
                    normalized_tag: str = self._normalize_text(tag)
                    if not normalized_tag:
                        continue
                    
                    # For multi-word tags, match the entire phrase with word boundaries
                    # For single-word tags, use word boundary to prevent false positives
                    if ' ' in normalized_tag:
                        # Multi-word tag: match entire phrase with word boundaries on both ends
                        # Escape the tag and replace spaces with \s+ to handle multiple spaces
                        escaped_tag: str = re.escape(normalized_tag).replace(r'\ ', r'\s+')
                        pattern: str = r'\b' + escaped_tag + r'\b'
                    else:
                        # Single-word tag: word boundary on both sides
                        pattern: str = r'\b' + re.escape(normalized_tag) + r'\b'
                    
                    if re.search(pattern, normalized_content):
                        found[tag] = self.vlm_detected_tag_confidence
        
        # Hallucinations (tags not in configured list) are automatically ignored
        return found

