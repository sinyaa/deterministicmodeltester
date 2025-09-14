from __future__ import annotations
from typing import Optional
from .base import BaseConnector, GenerateResult

class ReverseEchoConnector(BaseConnector):
    """Mock connector that returns the input text reversed."""
    
    def __init__(self, model_name: str = "reverse-echo", endpoint: str = "mock://reverse-echo", 
                 api_key: str = "", timeout_s: int = 30, max_retries: int = 3, retry_delay: float = 10.0):
        super().__init__(model_name, endpoint, api_key, timeout_s, max_retries, retry_delay)
    
    def _generate_single(self, prompt: str) -> GenerateResult:
        """Generate response by returning the input text reversed."""
        try:
            # Return the text reversed
            response_text = prompt[::-1]
            
            return GenerateResult(
                text=response_text,
                model=self.model_name,
                error=None
            )
        except Exception as e:
            return GenerateResult(
                text="",
                model=self.model_name,
                error=f"ReverseEcho generation failed: {str(e)}"
            )
