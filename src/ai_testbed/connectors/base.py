from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class GenerateResult:
    text: str
    model: str

class BaseConnector(ABC):
    """Common interface for all model connectors."""

    def __init__(self, model_name: str, endpoint: str, api_key: str, timeout_s: int = 30) -> None:
        self.model_name = model_name
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_s = timeout_s

    @abstractmethod
    def generate(self, prompt: str) -> GenerateResult:
        """Synchronous single-turn completion. Keep simple for bootstrap."""
        raise NotImplementedError
