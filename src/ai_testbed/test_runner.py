from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import colorama
from colorama import Fore, Back, Style
from .config.loader import load_app_config, load_test_config, load_test_run_config, TestConfig
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
                 tests_config_path: str = "config/tests.yaml",
                 test_run_config_path: str = "config/test-run.yaml"):
        self.models_config = load_app_config(models_config_path)
        self.tests_config = load_test_config(tests_config_path)
        self.test_run_config = load_test_run_config(test_run_config_path)
    
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
        
        # Check if model exists in models config
        if model_name not in self.models_config.models:
            return TestResult(
                test_name=test_name,
                model_name=model_name,
                passed=False,
                expected=test_config.expected_output,
                actual="",
                run_number=run_number,
                error=f"Model '{model_name}' not found in models configuration",
                distance=None
            )
        
        # Log test start
        run_info = f" (Run {run_number})" if run_number > 1 else ""
        print(f"  {Fore.CYAN}→{Style.RESET_ALL} Running {Fore.YELLOW}{test_name}{Style.RESET_ALL} on {Fore.BLUE}{model_name}{Style.RESET_ALL}{run_info}...", end=" ", flush=True)
        
        start_time = time.time()
        
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
                passed = test_config.expected_output.lower() in actual_output.lower()
                distance = None  # No distance calculation for substring matches
            
            # Log test result
            elapsed_time = time.time() - start_time
            if passed:
                print(f"{Fore.GREEN}✅ PASS{Style.RESET_ALL} ({elapsed_time:.2f}s)")
            else:
                print(f"{Fore.RED}❌ FAIL{Style.RESET_ALL} ({elapsed_time:.2f}s)")
            
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
            elapsed_time = time.time() - start_time
            print(f"{Fore.RED}❌ ERROR{Style.RESET_ALL} ({elapsed_time:.2f}s)")
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
        
        results = []
        
        # Run this test against all configured models
        for model_run in self.test_run_config.models:
            model_name = model_run.name
            # Use model-specific runs if specified, otherwise use runs_per_test
            runs = model_run.runs if hasattr(model_run, 'runs') and model_run.runs > 1 else self.test_run_config.runs_per_test
            
            for run_num in range(1, runs + 1):
                result = self.run_single_test(test_name, model_name, run_num)
                results.append(result)
        
        return results
    
    def run_all_tests(self) -> Dict[str, List[TestResult]]:
        """Run all tests against their configured models."""
        all_results = {}
        
        # Get all test names from tests.yaml
        test_names = list(self.tests_config.tests.keys())
        
        # Calculate total number of test runs
        total_runs = len(test_names) * sum(model_run.runs for model_run in self.test_run_config.models)
        
        print(f"\n{Fore.CYAN}🚀 Starting test execution...{Style.RESET_ALL}")
        print(f"   {Fore.YELLOW}Tests:{Style.RESET_ALL} {len(test_names)}")
        print(f"   {Fore.YELLOW}Total Runs:{Style.RESET_ALL} {total_runs}")
        print(f"   {Fore.YELLOW}Models:{Style.RESET_ALL} {len(self.test_run_config.models)}")
        print()
        
        for test_name in test_names:
            print(f"{Fore.YELLOW}📋 Test: {test_name}{Style.RESET_ALL}")
            print("-" * 60)
            
            results = self.run_test(test_name)
            all_results[test_name] = results
            
            # Count runs for this test
            test_runs = len(results)
            
            # Show test summary
            passed = sum(1 for r in results if r.passed)
            print(f"   {Fore.CYAN}Test Summary:{Style.RESET_ALL} {passed}/{test_runs} passed")
            print()
        
        return all_results
    
    def _print_model_comparison_table(self, results: Dict[str, List[TestResult]]) -> None:
        """Print a comparison table showing model performance with distance-based scoring."""
        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}MODEL COMPARISON TABLE{Style.RESET_ALL}")
        print("=" * 80)
        
        # Collect all model statistics with distance-based scoring
        model_stats = {}
        test_names = list(results.keys())
        
        for test_name, test_results in results.items():
            # Group results by model
            model_results = {}
            for result in test_results:
                if result.model_name not in model_results:
                    model_results[result.model_name] = []
                model_results[result.model_name].append(result)
            
            # Calculate stats for each model in this test
            for model_name, model_test_results in model_results.items():
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        'total_tests': 0,
                        'passed_tests': 0,
                        'total_distance': 0,
                        'avg_distance': 0,
                        'score': 0,
                        'test_results': {}
                    }
                
                model_passed = sum(1 for r in model_test_results if r.passed)
                model_total = len(model_test_results)
                
                # Calculate distance-based metrics
                total_distance = sum(r.distance or 0 for r in model_test_results)
                avg_distance = total_distance / model_total if model_total > 0 else 0
                
                # Calculate score: pass rate weighted by distance (lower distance = better score)
                # Score = pass_rate - (avg_distance / 100) to penalize high distances
                pass_rate = (model_passed / model_total * 100) if model_total > 0 else 0
                distance_penalty = min(avg_distance / 100, 50)  # Cap penalty at 50%
                score = max(pass_rate - distance_penalty, 0)  # Ensure non-negative
                
                # For exact match tests, if all tests fail, score should be based on distance
                if model_passed == 0 and avg_distance > 0:
                    # For failed tests, score inversely proportional to distance (closer = better)
                    # Max score for failed tests is 50, decreases as distance increases
                    score = max(50 - (avg_distance / 10), 0)
                
                model_stats[model_name]['total_tests'] += model_total
                model_stats[model_name]['passed_tests'] += model_passed
                model_stats[model_name]['total_distance'] += total_distance
                model_stats[model_name]['test_results'][test_name] = {
                    'passed': model_passed,
                    'total': model_total,
                    'pass_rate': pass_rate,
                    'avg_distance': avg_distance,
                    'score': score
                }
        
        # Calculate overall scores for each model
        for model_name, stats in model_stats.items():
            overall_pass_rate = (stats['passed_tests'] / stats['total_tests'] * 100) if stats['total_tests'] > 0 else 0
            overall_avg_distance = (stats['total_distance'] / stats['total_tests']) if stats['total_tests'] > 0 else 0
            
            # Overall score calculation
            if overall_pass_rate == 100:
                # Perfect pass rate
                overall_score = 100
            elif overall_pass_rate > 0:
                # Partial pass rate with distance penalty
                distance_penalty = min(overall_avg_distance / 100, 50)
                overall_score = max(overall_pass_rate - distance_penalty, 0)
            else:
                # All tests failed - score based on distance (closer = better)
                overall_score = max(50 - (overall_avg_distance / 10), 0)
            
            stats['avg_distance'] = overall_avg_distance
            stats['score'] = overall_score
        
        # Sort models by score (descending)
        sorted_models = sorted(
            model_stats.items(), 
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        # Print table header
        print(f"\n{Fore.YELLOW}{'Model':<15} {'Score':<8} {'Pass%':<8} {'Avg Dist':<10} {'Tests':<8}{Style.RESET_ALL}")
        print("-" * (15 + 8 + 8 + 10 + 8))
        
        # Print each model's row
        for rank, (model_name, stats) in enumerate(sorted_models, 1):
            overall_pass_rate = (stats['passed_tests'] / stats['total_tests'] * 100) if stats['total_tests'] > 0 else 0
            score = stats['score']
            avg_distance = stats['avg_distance']
            
            # Color coding for score
            if score >= 80:
                score_color = Fore.GREEN
                rank_symbol = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            elif score >= 50:
                score_color = Fore.YELLOW
                rank_symbol = "  "
            else:
                score_color = Fore.RED
                rank_symbol = "  "
            
            # Distance color coding
            if avg_distance <= 5:
                distance_color = Fore.GREEN
            elif avg_distance <= 20:
                distance_color = Fore.YELLOW
            else:
                distance_color = Fore.RED
            
            print(f"{rank_symbol} {Fore.BLUE}{model_name:<13}{Style.RESET_ALL} {score_color}{score:>6.1f}{Style.RESET_ALL} {overall_pass_rate:>6.1f}% {distance_color}{avg_distance:>8.1f}{Style.RESET_ALL} {stats['passed_tests']:>3}/{stats['total_tests']:<3}")
        
        print("-" * (15 + 8 + 8 + 10 + 8))
    
    def _print_test_model_matrix(self, results: Dict[str, List[TestResult]]) -> None:
        """Print a test-model matrix showing pass/fail counts for each combination."""
        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}TEST-MODEL MATRIX{Style.RESET_ALL}")
        print("=" * 80)
        
        # Get all unique models from results
        all_models = set()
        for test_results in results.values():
            for result in test_results:
                all_models.add(result.model_name)
        
        all_models = sorted(list(all_models))
        
        # Get all test names
        test_names = sorted(results.keys())
        
        if not test_names or not all_models:
            print("No test results to display")
            return
        
        # Calculate column widths
        test_name_width = max(len(name) for name in test_names) + 2
        model_width = 12
        
        # Print header
        print(f"{'Test':<{test_name_width}}", end="")
        for model in all_models:
            print(f"{model:^{model_width}}", end="")
        print()
        
        # Print separator line
        print("-" * test_name_width + "".join("-" * model_width for _ in all_models))
        
        # Print each test row
        for test_name in test_names:
            test_results = results[test_name]
            print(f"{test_name:<{test_name_width}}", end="")
            
            for model in all_models:
                # Count passes and total for this test-model combination
                model_test_results = [r for r in test_results if r.model_name == model]
                if not model_test_results:
                    print(f"{'N/A':^{model_width}}", end="")
                else:
                    passed = sum(1 for r in model_test_results if r.passed)
                    total = len(model_test_results)
                    pass_rate = (passed / total * 100) if total > 0 else 0
                    
                    # Color code based on pass rate
                    if pass_rate >= 80:
                        color = Fore.GREEN
                    elif pass_rate >= 50:
                        color = Fore.YELLOW
                    else:
                        color = Fore.RED
                    
                    result_text = f"{passed}/{total} ({pass_rate:.0f}%)"
                    print(f"{color}{result_text:^{model_width}}{Style.RESET_ALL}", end="")
            
            print()
        
        print("-" * test_name_width + "".join("-" * model_width for _ in all_models))
    
    def print_results(self, results: Dict[str, List[TestResult]]) -> None:
        """Print test results with model comparison table and distance-based scoring."""
        print("=" * 80)
        print(f"{Fore.CYAN}MODEL TEST RESULTS{Style.RESET_ALL}")
        print("=" * 80)
        
        # Calculate total tests for summary
        total_tests = sum(len(test_results) for test_results in results.values())
        passed_tests = sum(1 for test_results in results.values() for result in test_results if result.passed)
        
        # Show detailed test execution summary
        print(f"\n{Fore.YELLOW}📊 Test Execution Summary:{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}Total Tests:{Style.RESET_ALL} {total_tests}")
        print(f"  {Fore.GREEN}Tests Passed:{Style.RESET_ALL} {passed_tests}")
        print(f"  {Fore.RED}Tests Failed:{Style.RESET_ALL} {total_tests - passed_tests}")
        
        # Calculate additional statistics
        failed_tests = total_tests - passed_tests
        if total_tests > 0:
            pass_rate = (passed_tests / total_tests * 100)
            print(f"  {Fore.CYAN}Success Rate:{Style.RESET_ALL} {pass_rate:.1f}%")
            
            # Show test distribution by type
            exact_match_tests = sum(1 for test_results in results.values() 
                                  for result in test_results 
                                  if result.distance is not None)
            substring_tests = total_tests - exact_match_tests
            
            print(f"  {Fore.CYAN}Test Types:{Style.RESET_ALL} {exact_match_tests} exact match, {substring_tests} substring")
            
            # Show error statistics
            error_tests = sum(1 for test_results in results.values() 
                            for result in test_results 
                            if result.error is not None)
            if error_tests > 0:
                print(f"  {Fore.RED}Errors:{Style.RESET_ALL} {error_tests} tests had errors")
        
        # Show failed test details
        failed_tests = [result for test_results in results.values() for result in test_results if not result.passed]
        if failed_tests:
            print(f"\n{Fore.RED}❌ Failed Test Details:{Style.RESET_ALL}")
            print("-" * 80)
            
            for result in failed_tests:
                test_config = self.tests_config.tests.get(result.test_name)
                if test_config and test_config.exact_match and result.distance is not None:
                    print(f"{Fore.RED}Test:{Style.RESET_ALL} {result.test_name} | {Fore.RED}Model:{Style.RESET_ALL} {result.model_name} | {Fore.RED}Run:{Style.RESET_ALL} {result.run_number}")
                    print(f"  {Fore.YELLOW}Distance:{Style.RESET_ALL} {result.distance}")
                    print(f"  {Fore.GREEN}Expected:{Style.RESET_ALL} {result.expected}")
                    print(f"  {Fore.RED}Received:{Style.RESET_ALL} {result.actual}")
                    if result.error:
                        print(f"  {Fore.RED}Error:{Style.RESET_ALL} {result.error}")
                    print()
                else:
                    print(f"{Fore.RED}Test:{Style.RESET_ALL} {result.test_name} | {Fore.RED}Model:{Style.RESET_ALL} {result.model_name} | {Fore.RED}Run:{Style.RESET_ALL} {result.run_number}")
                    print(f"  {Fore.GREEN}Expected:{Style.RESET_ALL} {result.expected}")
                    print(f"  {Fore.RED}Received:{Style.RESET_ALL} {result.actual}")
                    if result.error:
                        print(f"  {Fore.RED}Error:{Style.RESET_ALL} {result.error}")
                    print()
        
        # Test-Model matrix table
        self._print_test_model_matrix(results)
        
        # Model comparison table with distance-based scoring
        self._print_model_comparison_table(results)
        
