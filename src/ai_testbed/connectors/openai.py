from __future__ import annotations
import requests
import json
from typing import Optional
from .base import BaseConnector, GenerateResult


class OpenAIConnector(BaseConnector):
    """OpenAI API connector for GPT models."""
    
    def _should_retry_empty_response(self, result: GenerateResult, attempt: int) -> bool:
        """Retry on empty responses for OpenAI API calls."""
        return not result.text or result.text.strip() == ""

    def _generate_single(self, prompt: str) -> GenerateResult:
        """Single generation attempt using the OpenAI API."""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Check if this is the responses API endpoint
            if "/v1/responses" in self.endpoint:
                # Responses API format - simplified parameters
                payload = {
                    "model": self.model_name,
                    "input": prompt
                }
            else:
                # Chat completions API format
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
            
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_s
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if this is the responses API format
                if "/v1/responses" in self.endpoint:
                    # Responses API format - extract text from the complex response
                    content = ""
                    if "output" in data and isinstance(data["output"], list):
                        for item in data["output"]:
                            if item.get("type") == "message" and "content" in item:
                                for content_item in item["content"]:
                                    if content_item.get("type") == "output_text":
                                        content = content_item.get("text", "")
                                        break
                                if content:
                                    break
                else:
                    # Chat completions API format
                    content = data["choices"][0]["message"]["content"]
                
                return GenerateResult(text=content, model=self.model_name)
            else:
                # Include rate limit information in error message
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                if response.status_code == 429:
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        error_msg += f" (retry-after: {retry_after} seconds)"
                return GenerateResult(text="", model=self.model_name, error=error_msg)
                
        except requests.exceptions.Timeout:
            return GenerateResult(text="", model=self.model_name, error="Request timeout")
        except requests.exceptions.RequestException as e:
            return GenerateResult(text="", model=self.model_name, error=f"Request error: {str(e)}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return GenerateResult(text="", model=self.model_name, error=f"Response parsing error: {str(e)}")
        except Exception as e:
            return GenerateResult(text="", model=self.model_name, error=f"Unexpected error: {str(e)}")
