from __future__ import annotations
import requests
import json
from typing import Optional
from .base import BaseConnector, GenerateResult


class AnthropicConnector(BaseConnector):
    """Anthropic API connector for Claude models."""
    
    def _should_retry_empty_response(self, result: GenerateResult, attempt: int) -> bool:
        """Retry on empty responses for Anthropic API calls."""
        return not result.text or result.text.strip() == ""

    def _generate_single(self, prompt: str) -> GenerateResult:
        """Single generation attempt using the Anthropic API."""
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": self.model_name,
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_s
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract content from Anthropic response format
                content = ""
                if "content" in data and len(data["content"]) > 0:
                    content = data["content"][0].get("text", "")
                
                return GenerateResult(text=content, model=self.model_name)
            else:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                return GenerateResult(text="", model=self.model_name, error=error_msg)
                
        except requests.exceptions.Timeout:
            return GenerateResult(text="", model=self.model_name, error="Request timeout")
        except requests.exceptions.RequestException as e:
            return GenerateResult(text="", model=self.model_name, error=f"Request error: {str(e)}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return GenerateResult(text="", model=self.model_name, error=f"Response parsing error: {str(e)}")
        except Exception as e:
            return GenerateResult(text="", model=self.model_name, error=f"Unexpected error: {str(e)}")
