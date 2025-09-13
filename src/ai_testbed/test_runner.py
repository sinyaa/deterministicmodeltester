from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import colorama
from colorama import Fore, Back, Style
from .config.loader import load_app_config, load_test_config, TestConfig
from .connectors.registry import create_connector

# Initialize colorama for cross-platform color support
colorama.init()

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

@dataclass
class TestResult:
    test_name: str
    model_name: str
    passed: bool
    expected: str
    actual: str
    run_number: int = 1
    error: Optional[str] = None
    distance: Optional[int] = None  # Lexicographical distance for exact match tests

class ModelTestRunner:
    """Runs model tests defined in YAML configuration files."""
    
    def __init__(self, models_config_path: str = "config/models.yaml", 
                 tests_config_path: str = "config/tests.yaml"):
        self.models_config = load_app_config(models_config_path)
        self.tests_config = load_test_config(tests_config_path)
    
    def run_single_test(self, test_name: str, model_name: str, run_number: int = 1) -> TestResult:
        """Run a single test against a specific model."""
        if test_name not in self.tests_config.tests:
            return TestResult(
                test_name=test_name,
                model_name=model_name,
                passed=False,
                expected="",
                actual="",
                run_number=run_number,
                error=f"Test '{test_name}' not found in configuration",
                distance=None
            )
        
        test_config = self.tests_config.tests[test_name]
        
        if model_name not in test_config.models:
            return TestResult(
                test_name=test_name,
                model_name=model_name,
                passed=False,
                expected=test_config.expected_output,
                actual="",
                run_number=run_number,
                error=f"Model '{model_name}' not configured for test '{test_name}'",
                distance=None
            )
        
        try:
            # Create connector for the model
            connector = create_connector(model_name, self.models_config)
            
            # Generate response
            result = connector.generate(test_config.prompt)
            actual_output = result.text
            
            # Check if test passes
            if test_config.exact_match:
                expected_stripped = test_config.expected_output.strip()
                actual_stripped = actual_output.strip()
                passed = expected_stripped == actual_stripped
                # Calculate lexicographical distance for exact match tests
                distance = levenshtein_distance(expected_stripped, actual_stripped)
            else:
                passed = test_config.expected_output in actual_output
                distance = None  # No distance calculation for substring matches
            
            return TestResult(
                test_name=test_name,
                model_name=model_name,
                passed=passed,
                expected=test_config.expected_output,
                actual=actual_output,
                run_number=run_number,
                error=None,
                distance=distance
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                model_name=model_name,
                passed=False,
                expected=test_config.expected_output,
                actual="",
                run_number=run_number,
                error=str(e),
                distance=None
            )
    
    def run_test(self, test_name: str) -> List[TestResult]:
        """Run a test against all configured models for that test."""
        if test_name not in self.tests_config.tests:
            return []
        
        test_config = self.tests_config.tests[test_name]
        results = []
        runs_per_test = getattr(self.tests_config, 'runs_per_test', 1)
        
        for model_name in test_config.models:
            for run_num in range(1, runs_per_test + 1):
                result = self.run_single_test(test_name, model_name, run_num)
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
        """Print test results in a formatted way with color coding and pass rates."""
        print("=" * 80)
        print(f"{Fore.CYAN}MODEL TEST RESULTS{Style.RESET_ALL}")
        print("=" * 80)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, test_results in results.items():
            print(f"\n{Fore.YELLOW}Test: {test_name}{Style.RESET_ALL}")
            print("-" * 40)
            
            # Group results by model for better display
            model_results = {}
            for result in test_results:
                if result.model_name not in model_results:
                    model_results[result.model_name] = []
                model_results[result.model_name].append(result)
            
            for model_name, model_test_results in model_results.items():
                print(f"\n  {Fore.BLUE}Model: {model_name}{Style.RESET_ALL}")
                
                # Calculate pass rate for this model
                model_passed = sum(1 for r in model_test_results if r.passed)
                model_total = len(model_test_results)
                model_pass_rate = (model_passed / model_total * 100) if model_total > 0 else 0
                
                for result in model_test_results:
                    total_tests += 1
                    if result.passed:
                        passed_tests += 1
                        status = f"{Fore.GREEN}✅ PASS{Style.RESET_ALL}"
                    else:
                        status = f"{Fore.RED}❌ FAIL{Style.RESET_ALL}"
                    
                    run_info = f" (Run {result.run_number})" if len(model_test_results) > 1 else ""
                    print(f"    {status}{run_info}")
                    
                    if not result.passed:
                        if result.error:
                            print(f"      {Fore.RED}Error: {result.error}{Style.RESET_ALL}")
                        else:
                            print(f"      {Fore.RED}Expected: {result.expected[:100]}{'...' if len(result.expected) > 100 else ''}{Style.RESET_ALL}")
                            print(f"      {Fore.RED}Actual:   {result.actual[:100]}{'...' if len(result.actual) > 100 else ''}{Style.RESET_ALL}")
                            # Show lexicographical distance for exact match tests
                            if result.distance is not None:
                                distance_color = Fore.GREEN if result.distance <= 5 else Fore.YELLOW if result.distance <= 20 else Fore.RED
                                print(f"      {distance_color}Distance: {result.distance} characters{Style.RESET_ALL}")
                
                # Show pass rate for this model
                if len(model_test_results) > 1:
                    pass_rate_color = Fore.GREEN if model_pass_rate >= 80 else Fore.YELLOW if model_pass_rate >= 50 else Fore.RED
                    print(f"    {pass_rate_color}Pass Rate: {model_pass_rate:.1f}% ({model_passed}/{model_total}){Style.RESET_ALL}")
        
        # Overall summary
        overall_pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        pass_rate_color = Fore.GREEN if overall_pass_rate >= 80 else Fore.YELLOW if overall_pass_rate >= 50 else Fore.RED
        
        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}SUMMARY:{Style.RESET_ALL} {passed_tests}/{total_tests} tests passed")
        print(f"{pass_rate_color}Overall Pass Rate: {overall_pass_rate:.1f}%{Style.RESET_ALL}")
        print("=" * 80)
