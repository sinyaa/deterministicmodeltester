from ai_testbed.harness.test_harness import TestHarness
from ai_testbed.models.types import TestCase

def test_harness_with_mock_model(tmp_path):
    # Use repo config by default
    h = TestHarness()
    case = TestCase(
        model="mock-gpt",
        prompt="hello",
        expected_contains="olleh",  # reversed by mock connector
    )
    ok, text = h.run_case(case)
    assert ok, f"Expected substring not found in: {text}"
    assert text.startswith("MOCK[mock-gpt]::")
