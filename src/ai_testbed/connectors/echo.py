from __future__ import annotations
from .base import BaseConnector, GenerateResult


class EchoConnector(BaseConnector):
    """Echo connector that simply returns the input prompt as output."""
    
    def _generate_single(self, prompt: str) -> GenerateResult:
        """Echo the input prompt back as the output."""
        return GenerateResult(
            text=prompt,
            model=self.model_name,
            error=None
        )
