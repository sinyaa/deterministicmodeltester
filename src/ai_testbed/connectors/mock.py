from __future__ import annotations
from .base import BaseConnector, GenerateResult

class MockConnector(BaseConnector):
    """Deterministic stub to simulate a model call without network."""

    def _generate_single(self, prompt: str) -> GenerateResult:
        # Different behavior based on model name for testing
        if self.model_name == "mock-gpt-2":
            # Echo the full input text
            reply = prompt
        else:
            # Return first 10 characters of input text for other models
            reply = prompt[:10] if len(prompt) >= 10 else prompt
        return GenerateResult(text=reply, model=self.model_name, error=None)
