from __future__ import annotations
from typing import Optional
from .base import BaseConnector, GenerateResult

class HalfEchoConnector(BaseConnector):
    """Mock connector that returns the first half of the input text."""
    
    def __init__(self, model_name: str = "half-echo", endpoint: str = "mock://half-echo", 
                 api_key: str = "", timeout_s: int = 30, max_retries: int = 3, retry_delay: float = 10.0):
        super().__init__(model_name, endpoint, api_key, timeout_s, max_retries, retry_delay)
    
    def _generate_single(self, prompt: str) -> GenerateResult:
        """Generate response by returning the first half of the input text."""
        try:
            # Calculate the midpoint
            text_length = len(prompt)
            half_length = text_length // 2
            
            # Return the first half of the text
            response_text = prompt[:half_length]
            
            return GenerateResult(
                text=response_text,
                model=self.model_name,
                error=None
            )
        except Exception as e:
            return GenerateResult(
                text="",
                model=self.model_name,
                error=f"HalfEcho generation failed: {str(e)}"
            )
