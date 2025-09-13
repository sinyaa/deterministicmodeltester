from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .config.loader import load_app_config, load_test_config, TestConfig
from .connectors.registry import create_connector

@dataclass
class TestResult:
    test_name: str
    model_name: str
    passed: bool
    expected: str
    actual: str
    error: Optional[str] = None

class ModelTestRunner:
    """Runs model tests defined in YAML configuration files."""
    
    def __init__(self, models_config_path: str = "config/models.yaml", 
                 tests_config_path: str = "config/tests.yaml"):
        self.models_config = load_app_config(models_config_path)
        self.tests_config = load_test_config(tests_config_path)
    
    def run_single_test(self, test_name: str, model_name: str) -> TestResult:
        """Run a single test against a specific model."""
        if test_name not in self.tests_config.tests:
            return TestResult(
                test_name=test_name,
                model_name=model_name,
                passed=False,
                expected="",
                actual="",
                error=f"Test '{test_name}' not found in configuration"
            )
        
        test_config = self.tests_config.tests[test_name]
        
        if model_name not in test_config.models:
            return TestResult(
                test_name=test_name,
                model_name=model_name,
                passed=False,
                expected=test_config.expected_output,
                actual="",
                error=f"Model '{model_name}' not configured for test '{test_name}'"
            )
        
        try:
            # Create connector for the model
            connector = create_connector(model_name, self.models_config)
            
            # Generate response
            result = connector.generate(test_config.prompt)
            actual_output = result.text
            
            # Check if test passes
            if test_config.exact_match:
                passed = actual_output.strip() == test_config.expected_output.strip()
            else:
                passed = test_config.expected_output in actual_output
            
            return TestResult(
                test_name=test_name,
                model_name=model_name,
                passed=passed,
                expected=test_config.expected_output,
                actual=actual_output,
                error=None
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                model_name=model_name,
                passed=False,
                expected=test_config.expected_output,
                actual="",
                error=str(e)
            )
    
    def run_test(self, test_name: str) -> List[TestResult]:
        """Run a test against all configured models for that test."""
        if test_name not in self.tests_config.tests:
            return []
        
        test_config = self.tests_config.tests[test_name]
        results = []
        
        for model_name in test_config.models:
            result = self.run_single_test(test_name, model_name)
            results.append(result)
        
        return results
    
    def run_all_tests(self) -> Dict[str, List[TestResult]]:
        """Run all tests against their configured models."""
        all_results = {}
        
        for test_name in self.tests_config.tests:
            results = self.run_test(test_name)
            all_results[test_name] = results
        
        return all_results
    
    def print_results(self, results: Dict[str, List[TestResult]]) -> None:
        """Print test results in a formatted way."""
        print("=" * 80)
        print("MODEL TEST RESULTS")
        print("=" * 80)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, test_results in results.items():
            print(f"\nTest: {test_name}")
            print("-" * 40)
            
            for result in test_results:
                total_tests += 1
                if result.passed:
                    passed_tests += 1
                    status = "✅ PASS"
                else:
                    status = "❌ FAIL"
                
                print(f"  {result.model_name}: {status}")
                
                if not result.passed:
                    if result.error:
                        print(f"    Error: {result.error}")
                    else:
                        print(f"    Expected: {result.expected[:100]}{'...' if len(result.expected) > 100 else ''}")
                        print(f"    Actual:   {result.actual[:100]}{'...' if len(result.actual) > 100 else ''}")
        
        print("\n" + "=" * 80)
        print(f"SUMMARY: {passed_tests}/{total_tests} tests passed")
        print("=" * 80)
