from __future__ import annotations
from .base import BaseConnector, GenerateResult

class MockConnector(BaseConnector):
    """Deterministic stub to simulate a model call without network."""

    def generate(self, prompt: str) -> GenerateResult:
        # Simple, predictable behavior for tests: reverse string and prefix with model name.
        reply = f"MOCK[{self.model_name}]::" + prompt[::-1]
        return GenerateResult(text=reply, model=self.model_name)
