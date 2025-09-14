from ai_testbed.harness.test_harness import TestHarness
from ai_testbed.models.types import TestCase

def test_harness_with_mock_model(tmp_path):
    # Use repo config by default
    h = TestHarness()
    case = TestCase(
        model="mock-gpt",
        prompt="hello",
        expected_contains="hello",  # Mock connector now returns first 10 characters
    )
    ok, text = h.run_case(case)
    assert ok, f"Expected substring not found in: {text}"
    assert text == "hello"  # Mock connector now returns text directly
