"""
Comprehensive Unit Tests for BaseVLMClient
Provides 100% test coverage for prompt generation and response parsing.
"""

import pytest
import logging
from typing import Dict, List, Any
from unittest.mock import Mock, patch

# Import the module under test
from vlm_engine.base_vlm_client import BaseVLMClient


class TestBaseVLMClientPromptGeneration:
    """Test cases for prompt generation in BaseVLMClient"""
    
    def test_prompt_generation_basic(self):
        """Test basic prompt generation with small tag list"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1", "tag2", "tag3"]
        }
        client = BaseVLMClient(config)
        prompt: str = client._build_prompt_text()
        
        # Verify prompt contains key elements
        assert "Focus ONLY on what you actually see" in prompt
        assert "MAX 15 WORDS" in prompt
        assert "tag1, tag2, tag3" in prompt
        assert "EXACTLY match" in prompt
        assert "none" in prompt
        assert "|" in prompt  # Delimiter should be present
        
        # Verify example is removed
        assert "Example:" not in prompt
        assert "dog" not in prompt.lower()
        assert "sitting" not in prompt.lower()
    
    def test_prompt_generation_large_tag_list(self):
        """Test prompt generation with large tag list (>30 tags)"""
        large_tag_list: List[str] = [f"tag{i}" for i in range(35)]
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": large_tag_list
        }
        client = BaseVLMClient(config)
        prompt: str = client._build_prompt_text()
        
        # Verify all tags are included
        assert "tag0" in prompt
        assert "tag34" in prompt
        assert "MAX 15 WORDS" in prompt
        assert "EXACTLY match" in prompt
    
    def test_prompt_generation_with_special_tokens(self):
        """Test prompt generation when special tokens are configured"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1", "tag2"],
            "special_tokens": {
                "begin": "<|begin|>",
                "end": "<|end|>"
            }
        }
        client = BaseVLMClient(config)
        prompt: str = client._build_prompt_text()
        
        # Verify special tokens are mentioned in output format
        assert "<|begin|>" in prompt
        assert "<|end|>" in prompt
    
    def test_prompt_generation_word_limit_emphasis(self):
        """Test that word limit is emphasized in prompt"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1"]
        }
        client = BaseVLMClient(config)
        prompt: str = client._build_prompt_text()
        
        # Verify word limit appears multiple times
        assert prompt.count("MAX 15 WORDS") >= 2
    
    def test_prompt_generation_exact_match_emphasis(self):
        """Test that exact matching is emphasized"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1"]
        }
        client = BaseVLMClient(config)
        prompt: str = client._build_prompt_text()
        
        # Verify exact matching instructions
        assert "EXACTLY match" in prompt
        assert "do not apply tags to similar but different content" in prompt.lower()
    
    def test_prompt_generation_none_handling(self):
        """Test that 'none' handling is clearly explained"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1"]
        }
        client = BaseVLMClient(config)
        prompt: str = client._build_prompt_text()
        
        # Verify none handling instructions
        assert "If no tags match" in prompt
        assert "[description] | none" in prompt


class TestBaseVLMClientParsing:
    """Test cases for response parsing in BaseVLMClient"""
    
    def test_parse_empty_response(self):
        """Test parsing empty response"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1", "tag2", "tag3"]
        }
        client = BaseVLMClient(config)
        result: Dict[str, float] = client._parse_simple_default("")
        
        assert result == {"tag1": 0.0, "tag2": 0.0, "tag3": 0.0}
    
    def test_parse_whitespace_only_response(self):
        """Test parsing whitespace-only response"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1", "tag2"]
        }
        client = BaseVLMClient(config)
        result: Dict[str, float] = client._parse_simple_default("   \n\t  ")
        
        assert result == {"tag1": 0.0, "tag2": 0.0}
    
    def test_parse_none_as_entire_reply(self):
        """Test parsing 'none' as the entire reply"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1", "tag2"],
            "vlm_detected_tag_confidence": 0.99
        }
        client = BaseVLMClient(config)
        result: Dict[str, float] = client._parse_simple_default("none")
        
        assert result == {"tag1": 0.0, "tag2": 0.0}
    
    def test_parse_none_after_delimiter(self):
        """Test parsing 'none' after the | delimiter (new format)"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1", "tag2"],
            "vlm_detected_tag_confidence": 0.99
        }
        client = BaseVLMClient(config)
        result: Dict[str, float] = client._parse_simple_default("A person sitting | none")
        
        assert result == {"tag1": 0.0, "tag2": 0.0}
    
    def test_parse_none_variations_after_delimiter(self):
        """Test parsing various 'none' variations after delimiter"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1", "tag2"],
            "vlm_detected_tag_confidence": 0.99
        }
        client = BaseVLMClient(config)
        
        test_cases: List[str] = [
            "Description here | nothing",
            "Description | no tags",
            "Description | n/a",
            "Description | na"
        ]
        
        for reply in test_cases:
            result: Dict[str, float] = client._parse_simple_default(reply)
            assert result == {"tag1": 0.0, "tag2": 0.0}, f"Failed for: {reply}"
    
    def test_parse_single_tag_after_delimiter(self):
        """Test parsing single tag after delimiter"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1", "tag2", "tag3"],
            "vlm_detected_tag_confidence": 0.99
        }
        client = BaseVLMClient(config)
        result: Dict[str, float] = client._parse_simple_default("Description | tag1")
        
        assert result == {"tag1": 0.99, "tag2": 0.0, "tag3": 0.0}
    
    def test_parse_multiple_tags_after_delimiter(self):
        """Test parsing multiple tags after delimiter"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1", "tag2", "tag3"],
            "vlm_detected_tag_confidence": 0.99
        }
        client = BaseVLMClient(config)
        result: Dict[str, float] = client._parse_simple_default("Description | tag1, tag2")
        
        assert result == {"tag1": 0.99, "tag2": 0.99, "tag3": 0.0}
    
    def test_parse_tags_with_extra_whitespace(self):
        """Test parsing tags with extra whitespace"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1", "tag2"],
            "vlm_detected_tag_confidence": 0.99
        }
        client = BaseVLMClient(config)
        result: Dict[str, float] = client._parse_simple_default("Description |  tag1  ,  tag2  ")
        
        assert result == {"tag1": 0.99, "tag2": 0.99}
    
    def test_parse_case_insensitive_matching(self):
        """Test that tag matching is case-insensitive"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["Tag1", "TAG2", "tag3"],
            "vlm_detected_tag_confidence": 0.99
        }
        client = BaseVLMClient(config)
        result: Dict[str, float] = client._parse_simple_default("Description | tag1, TAG2, TaG3")
        
        assert result == {"Tag1": 0.99, "TAG2": 0.99, "tag3": 0.99}
    
    def test_parse_without_delimiter(self):
        """Test parsing response without delimiter (fallback to entire content)"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1", "tag2"],
            "vlm_detected_tag_confidence": 0.99
        }
        client = BaseVLMClient(config)
        result: Dict[str, float] = client._parse_simple_default("tag1, tag2")
        
        assert result == {"tag1": 0.99, "tag2": 0.99}
    
    def test_parse_hallucinated_tags_ignored(self):
        """Test that tags not in tag_list are ignored (hallucinations)"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1", "tag2"],
            "vlm_detected_tag_confidence": 0.99
        }
        client = BaseVLMClient(config)
        result: Dict[str, float] = client._parse_simple_default("Description | tag1, hallucinated_tag, tag2")
        
        # Only valid tags should be matched
        assert result == {"tag1": 0.99, "tag2": 0.99}
        assert "hallucinated_tag" not in result
    
    def test_parse_special_tokens_extraction(self):
        """Test parsing with special tokens configured"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1", "tag2"],
            "vlm_detected_tag_confidence": 0.99,
            "special_tokens": {
                "begin": "<|begin|>",
                "end": "<|end|>"
            }
        }
        client = BaseVLMClient(config)
        result: Dict[str, float] = client._parse_simple_default(
            "Some text <|begin|>Description | tag1<|end|> more text"
        )
        
        assert result == {"tag1": 0.99, "tag2": 0.0}
    
    def test_parse_fallback_normalization(self):
        """Test fallback normalization-based parsing when primary path fails"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag one", "tag-two"],
            "vlm_detected_tag_confidence": 0.99
        }
        client = BaseVLMClient(config)
        # Response without exact comma-separated format
        result: Dict[str, float] = client._parse_simple_default("I see tag one and tag-two in the image")
        
        # Fallback should find tags using normalization
        assert result["tag one"] == 0.99
        assert result["tag-two"] == 0.99
    
    def test_parse_multi_word_tags(self):
        """Test parsing multi-word tags"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag one", "tag two", "tag"],
            "vlm_detected_tag_confidence": 0.99
        }
        client = BaseVLMClient(config)
        result: Dict[str, float] = client._parse_simple_default("Description | tag one, tag two")
        
        assert result == {"tag one": 0.99, "tag two": 0.99, "tag": 0.0}


class TestBaseVLMClientUtilities:
    """Test cases for utility methods in BaseVLMClient"""
    
    def test_normalize_text(self):
        """Test text normalization"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1"]
        }
        client = BaseVLMClient(config)
        
        assert client._normalize_text("Hello, World!") == "hello world"
        assert client._normalize_text("  Multiple   Spaces  ") == "multiple spaces"
        assert client._normalize_text("") == ""
        assert client._normalize_text("UPPERCASE") == "uppercase"
    
    def test_format_tag_list_small(self):
        """Test tag list formatting for small lists"""
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": ["tag1", "tag2", "tag3"]
        }
        client = BaseVLMClient(config)
        formatted: str = client._format_tag_list_for_prompt()
        
        assert formatted == "tag1, tag2, tag3"
    
    def test_format_tag_list_large(self):
        """Test tag list formatting for large lists"""
        large_list: List[str] = [f"tag{i}" for i in range(35)]
        config: Dict[str, Any] = {
            "model_id": "test-model",
            "tag_list": large_list
        }
        client = BaseVLMClient(config)
        formatted: str = client._format_tag_list_for_prompt()
        
        # Should still be comma-separated
        assert "tag0" in formatted
        assert "tag34" in formatted
        assert "," in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

