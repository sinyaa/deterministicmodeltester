from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
import random
from typing import Optional

@dataclass
class GenerateResult:
    text: str
    model: str
    error: str = None

class BaseConnector(ABC):
    """Common interface for all model connectors."""

    def __init__(self, model_name: str, endpoint: str, api_key: str, timeout_s: int = 30, 
                 max_retries: int = 3, retry_delay: float = 10.0) -> None:
        self.model_name = model_name
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _should_retry(self, result: GenerateResult, attempt: int) -> bool:
        """Determine if we should retry based on the result and attempt number."""
        if attempt >= self.max_retries:
            return False
        
        # Retry if there's an error
        if result.error is not None:
            return True
        
        # Allow subclasses to override retry behavior for empty responses
        return self._should_retry_empty_response(result, attempt)
    
    def _should_retry_empty_response(self, result: GenerateResult, attempt: int) -> bool:
        """Override this method to customize retry behavior for empty responses."""
        # Default: don't retry on empty responses unless there's an error
        return False

    def _wait_before_retry(self, attempt: int) -> None:
        """Wait before retrying with exponential backoff and jitter."""
        if attempt <= 0:
            return
            
        # Exponential backoff: base_delay * (2^attempt) + jitter
        delay = self.retry_delay * (2 ** (attempt - 1))
        jitter = random.uniform(0, 1)  # Add up to 1 second of jitter
        total_delay = delay + jitter
        
        print(f"  ⏳ Retrying in {total_delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})...")
        time.sleep(total_delay)

    def generate_with_retry(self, prompt: str) -> GenerateResult:
        """Generate with retry logic for robustness."""
        last_result = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = self._generate_single(prompt)
                last_result = result
                
                # Check if we should retry
                if not self._should_retry(result, attempt):
                    if attempt > 0:
                        print(f"  ✅ Success after {attempt + 1} attempts")
                    return result
                
                # Wait before retry (except on last attempt)
                if attempt < self.max_retries:
                    self._wait_before_retry(attempt)
                    
            except Exception as e:
                last_result = GenerateResult(
                    text="", 
                    model=self.model_name, 
                    error=f"Unexpected error on attempt {attempt + 1}: {str(e)}"
                )
                
                if attempt < self.max_retries:
                    self._wait_before_retry(attempt)
        
        # All retries exhausted
        if last_result and last_result.error:
            last_result.error = f"Failed after {self.max_retries + 1} attempts. Last error: {last_result.error}"
        else:
            last_result = GenerateResult(
                text="", 
                model=self.model_name, 
                error=f"Failed after {self.max_retries + 1} attempts with empty responses"
            )
        
        return last_result

    @abstractmethod
    def _generate_single(self, prompt: str) -> GenerateResult:
        """Single generation attempt without retry logic."""
        raise NotImplementedError

    def generate(self, prompt: str) -> GenerateResult:
        """Public interface that uses retry logic."""
        return self.generate_with_retry(prompt)
