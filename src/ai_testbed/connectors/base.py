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
    error: Optional[str] = None

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
        
        # Validate API key for non-local providers
        self._validate_api_key()

    def _validate_api_key(self) -> None:
        """Validate that API key is present for non-local providers."""
        # Skip validation for local/mock providers
        if self._is_local_provider():
            return
            
        # Check if API key is missing or invalid
        if not self.api_key or self.api_key.strip() == "" or self.api_key in ["dummy", "test-key", "mock-key", "${OPENAI_API_KEY}", "${ANTHROPIC_API_KEY}"]:
            provider_name = self._get_provider_name()
            raise ValueError(
                f"❌ API key is missing or invalid for {provider_name} model '{self.model_name}'\n"
                f"   Please set the {self._get_env_var_name()} environment variable\n"
                f"   Example: $env:{self._get_env_var_name()}=\"your-api-key-here\""
            )

    def _is_local_provider(self) -> bool:
        """Check if this is a local provider that doesn't need API keys."""
        return (
            "mock://" in self.endpoint or 
            self.endpoint.startswith("mock://") or
            self.model_name.startswith("echo-") or
            self.model_name.startswith("mock-")
        )

    def _get_provider_name(self) -> str:
        """Get human-readable provider name."""
        if "openai" in self.endpoint:
            return "OpenAI"
        elif "anthropic" in self.endpoint:
            return "Anthropic"
        else:
            return "API"

    def _get_env_var_name(self) -> str:
        """Get the environment variable name for this provider."""
        if "openai" in self.endpoint:
            return "OPENAI_API_KEY"
        elif "anthropic" in self.endpoint:
            return "ANTHROPIC_API_KEY"
        else:
            return "API_KEY"

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

    def _wait_before_retry(self, attempt: int, rate_limit_delay: float = 0) -> None:
        """Wait before retrying with exponential backoff and jitter."""
        if attempt <= 0 and rate_limit_delay <= 0:
            return
            
        # Use rate limit delay if provided, otherwise use exponential backoff
        if rate_limit_delay > 0:
            base_delay = rate_limit_delay
            jitter = random.uniform(0, min(5, rate_limit_delay * 0.1))  # 10% jitter, max 5s
            print(f"  ⏳ Rate limited, waiting {rate_limit_delay:.1f}s + {jitter:.1f}s jitter...")
        else:
            # Exponential backoff: base_delay * (2^attempt) + jitter
            delay = self.retry_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0, min(10, delay * 0.2))  # 20% jitter, max 10s
            base_delay = delay
        
        total_delay = base_delay + jitter
        
        print(f"  ⏳ Retrying in {total_delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})...")
        time.sleep(total_delay)

    def generate_with_retry(self, prompt: str) -> GenerateResult:
        """Generate with retry logic for robustness."""
        last_result = None
        rate_limit_delay = 0
        
        for attempt in range(self.max_retries + 1):
            try:
                result = self._generate_single(prompt)
                last_result = result
                
                # Check for rate limiting
                if result.error and "429" in result.error:
                    # Extract retry-after from error message if available
                    if "retry-after" in result.error.lower():
                        try:
                            # Look for retry-after value in error message
                            import re
                            match = re.search(r'retry-after[:\s]+(\d+)', result.error, re.IGNORECASE)
                            if match:
                                rate_limit_delay = float(match.group(1))
                        except:
                            rate_limit_delay = 60  # Default 60 seconds
                    else:
                        rate_limit_delay = 60  # Default 60 seconds
                    
                    print(f"  🚫 Rate limited detected: {result.error}")
                
                # Check if we should retry
                if not self._should_retry(result, attempt):
                    if attempt > 0:
                        print(f"  ✅ Success after {attempt + 1} attempts")
                    return result
                
                # Wait before retry (except on last attempt)
                if attempt < self.max_retries:
                    self._wait_before_retry(attempt, rate_limit_delay)
                    rate_limit_delay = 0  # Reset after first retry
                    
            except Exception as e:
                last_result = GenerateResult(
                    text="",
                    model=self.model_name,
                    error=f"Unexpected error on attempt {attempt + 1}: {str(e)}"
                )
                
                if attempt < self.max_retries:
                    self._wait_before_retry(attempt, rate_limit_delay)
                    rate_limit_delay = 0  # Reset after first retry
        
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
