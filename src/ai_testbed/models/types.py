from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TestCase:
    model: str
    prompt: str
    expected_contains: str
