from __future__ import annotations
from typing import Iterable, Tuple, List
from ..config.loader import load_app_config
from ..connectors.registry import create_connector
from ..models.types import TestCase

class TestHarness:
    def __init__(self, config_path: str = "config/models.yaml") -> None:
        self.cfg = load_app_config(config_path)

    def run_case(self, case: TestCase) -> Tuple[bool, str]:
        connector = create_connector(case.model, self.cfg)
        result = connector.generate(case.prompt)
        ok = case.expected_contains in result.text
        return ok, result.text

    def run_cases(self, cases: Iterable[TestCase]) -> List[Tuple[TestCase, bool, str]]:
        outcomes = []
        for c in cases:
            ok, text = self.run_case(c)
            outcomes.append((c, ok, text))
        return outcomes
