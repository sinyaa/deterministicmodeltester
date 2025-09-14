#!/usr/bin/env python3

import sys
sys.path.append('src')

from ai_testbed.test_runner import ModelTestRunner, TestResult
from ai_testbed.config.loader import AppConfig, ModelConfig, TestSuiteConfig, TestConfig, TestRunConfig
from unittest.mock import patch

def test_simple():
    with patch('ai_testbed.test_runner.load_test_run_config') as mock_load_test_run, \
         patch('ai_testbed.test_runner.load_test_config') as mock_load_test, \
         patch('ai_testbed.test_runner.load_app_config') as mock_load_app:
        
        mock_models_config = AppConfig(
            models={
                "mock-gpt": ModelConfig(
                    provider="mock",
                    endpoint="http://mock.com",
                    api_key="test-key"
                )
            }
        )

        mock_tests_config = TestSuiteConfig(
            tests={
                "test1": TestConfig(
                    name="Test 1",
                    description="Test description",
                    prompt="Test prompt",
                    expected_output="Expected output",
                    exact_match=True
                )
            }
        )

        mock_test_run_config = TestRunConfig(
            test_runs=[
                {"test": "test1", "model": "mock-gpt", "runs": 3}
            ],
            runs_per_test=3
        )

        mock_load_app.return_value = mock_models_config
        mock_load_test.return_value = mock_tests_config
        mock_load_test_run.return_value = mock_test_run_config

        runner = ModelTestRunner("dummy", "dummy", "dummy")
        
        print("Models config:", runner.models_config)
        print("Tests config:", runner.tests_config)
        print("Test run config:", runner.test_run_config)

        # Create mock results with multiple runs
        results = {
            "test1": [
                TestResult("test1", "mock-gpt", True, "Expected", "Actual", 1),
                TestResult("test1", "mock-gpt", False, "Expected", "Different", 2),
                TestResult("test1", "mock-gpt", True, "Expected", "Actual", 3),
            ]
        }

        runner.print_results(results)

if __name__ == "__main__":
    test_simple()
