from ai_testbed.test_runner import ModelTestRunner, TestResult

def test_deterministic_behavior_test():
    """Test that the deterministic behavior test works correctly."""
    runner = ModelTestRunner()
    
    # Run the deterministic behavior test
    results = runner.run_test("deterministic_behavior")
    
    # Should have one result for mock-gpt
    assert len(results) == 1
    
    result = results[0]
    assert result.test_name == "deterministic_behavior"
    assert result.model_name == "mock-gpt"
    
    # The mock connector reverses the input, so it should fail the exact match test
    assert not result.passed
    assert "MOCK[mock-gpt]::" in result.actual
    assert result.expected.startswith("This is the test prompt")

def test_single_test_run():
    """Test running a single test against a specific model."""
    runner = ModelTestRunner()
    
    result = runner.run_single_test("deterministic_behavior", "mock-gpt")
    
    assert result.test_name == "deterministic_behavior"
    assert result.model_name == "mock-gpt"
    assert not result.passed  # Mock connector reverses input
    assert result.error is None

def test_nonexistent_test():
    """Test handling of non-existent test."""
    runner = ModelTestRunner()
    
    result = runner.run_single_test("nonexistent_test", "mock-gpt")
    
    assert not result.passed
    assert result.error is not None
    assert "not found" in result.error

def test_nonexistent_model():
    """Test handling of model not configured for test."""
    runner = ModelTestRunner()
    
    result = runner.run_single_test("deterministic_behavior", "nonexistent-model")
    
    assert not result.passed
    assert result.error is not None
    assert "not configured" in result.error
