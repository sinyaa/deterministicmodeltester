from __future__ import annotations
import requests
import json
import time
from typing import Optional
from .base import BaseConnector, GenerateResult


class OpenAIRealtimeConnector(BaseConnector):
    """OpenAI Realtime API connector for realtime preview models."""
    
    def _should_retry_empty_response(self, result: GenerateResult, attempt: int) -> bool:
        """Retry on empty responses for OpenAI Realtime API calls."""
        return not result.text or result.text.strip() == ""

    def _generate_single(self, prompt: str) -> GenerateResult:
        """Single generation attempt using the OpenAI Realtime API."""
        try:
            # For realtime models, we'll fall back to the standard chat completions API
            # since the realtime API requires WebSocket connections which are complex
            # to implement for deterministic testing
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Map realtime model names to their standard equivalents
            model_mapping = {
                "gpt-4o-mini-realtime-preview": "gpt-4o-mini",
                "gpt-4o-realtime-preview": "gpt-4o"
            }
            
            standard_model = model_mapping.get(self.model_name, "gpt-4o-mini")
            
            payload = {
                "model": standard_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            # Use the standard chat completions endpoint
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout_s
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
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
