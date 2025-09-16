from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import colorama
from colorama import Fore, Back, Style
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore, Lock
from datetime import datetime
from .config.loader import load_app_config, load_test_config, load_test_run_config, TestConfig
from .connectors.registry import create_connector

# Initialize colorama for cross-platform color support
colorama.init()

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text for comparison."""
    import re
    # Replace multiple whitespace characters with single space
    normalized = re.sub(r'\s+', ' ', text.strip())
    return normalized

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
    latency_ms: Optional[float] = None  # First response latency in milliseconds

class ModelTestRunner:
    """Runs model tests defined in YAML configuration files with parallel execution."""
    
    def __init__(self, models_config_path: str = "config/models.yaml", 
                 tests_config_path: str = "config/tests-cases.yaml",
                 test_run_config_path: str = "config/test-run.yaml",
                 max_workers: int = 5):
        self.models_config = load_app_config(models_config_path)
        self.test_run_config, self.tests_config = load_test_run_config(test_run_config_path)
        
        # If tests_config is None (no tests field in config), fall back to tests_config_path
        if self.tests_config is None and tests_config_path:
            self.tests_config = load_test_config(tests_config_path)
        
        self.max_workers = max_workers
        
        # Rate limiting controls for different API providers
        self.rate_limiters = {
            'openai': Semaphore(10),  # Max 10 concurrent OpenAI calls
            'anthropic': Semaphore(5),  # Max 5 concurrent Anthropic calls
            'echo': Semaphore(20),  # Local echo can handle more
            'mock': Semaphore(20),  # Mock can handle more
        }
        
        # Failed test logging
        self.log_file_path = self._get_log_file_path()
        self.log_lock = Lock()  # Thread-safe logging
        
        # HTML result export
        self.html_file_path = self._get_html_file_path()
    
    def _get_log_file_path(self) -> str:
        """Generate log file path with current date."""
        current_date = datetime.now().strftime("%Y%m%d")
        return f"test-run-failed-{current_date}.log"
    
    def _get_html_file_path(self) -> str:
        """Generate HTML file path with current date-time."""
        current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"test-run-result-{current_datetime}.html"
    
    def _log_failed_test(self, result: TestResult, total_runs: int) -> None:
        """Log a failed test to the log file in a thread-safe manner."""
        if not result.passed:
            with self.log_lock:
                try:
                    with open(self.log_file_path, 'a', encoding='utf-8') as f:
                        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, {result.model_name}, {result.test_name}, Run ({result.run_number}/{total_runs})\n")
                        f.write(f"Expected output: {result.expected}\n")
                        f.write(f"Received output: {result.actual}\n")
                        if result.distance is not None:
                            f.write(f"Distance: {result.distance}\n")
                        else:
                            f.write("Distance: N/A (substring match)\n")
                        if result.error:
                            f.write(f"Error: {result.error}\n")
                        f.write("\n")
                except Exception as e:
                    # Don't let logging errors break the test execution
                    print(f"Warning: Failed to write to log file: {e}")
    
    def _export_results_to_html(self, results: Dict[str, List[TestResult]]) -> None:
        """Export test results to HTML file with MODEL COMPARISON TABLE and TEST-MODEL MATRIX."""
        try:
            # Calculate summary statistics
            total_tests = sum(len(test_results) for test_results in results.values())
            passed_tests = sum(1 for test_results in results.values() for result in test_results if result.passed)
            failed_tests = total_tests - passed_tests
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            # Count test types
            exact_match_tests = sum(1 for test_results in results.values() for result in test_results if result.distance is not None)
            substring_tests = total_tests - exact_match_tests
            
            # Get all unique models and tests
            all_models = sorted(set(result.model_name for test_results in results.values() for result in test_results))
            all_tests = sorted(results.keys())
            
            # Calculate model statistics
            model_stats = {}
            for model in all_models:
                model_results = [result for test_results in results.values() for result in test_results if result.model_name == model]
                model_passed = sum(1 for result in model_results if result.passed)
                model_total = len(model_results)
                pass_rate = (model_passed / model_total * 100) if model_total > 0 else 0
                
                # Calculate average distance for exact match tests
                exact_match_results = [result for result in model_results if result.distance is not None]
                avg_distance = sum(result.distance for result in exact_match_results) / len(exact_match_results) if exact_match_results else 0
                
                # Calculate score
                distance_penalty = min(avg_distance / 100, 50)  # Cap penalty at 50%
                score = max(pass_rate - distance_penalty, 0)  # Ensure non-negative
                
                # For exact match tests, if all tests fail, score should be based on distance
                if model_passed == 0 and avg_distance > 0:
                    score = max(50 - (avg_distance / 10), 0)
                
                model_stats[model] = {
                    'passed': model_passed,
                    'total': model_total,
                    'pass_rate': pass_rate,
                    'avg_distance': avg_distance,
                    'score': score
                }
            
            # Sort models by score (descending)
            sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['score'], reverse=True)
            
            # Generate HTML content
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Model Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .summary {{
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .summary-item {{
            background: white;
            padding: 20px;
            border-radius: 6px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .summary-item h3 {{
            margin: 0 0 10px 0;
            color: #495057;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .summary-item .value {{
            font-size: 2em;
            font-weight: bold;
            color: #212529;
        }}
        .success {{ color: #28a745; }}
        .danger {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        .info {{ color: #17a2b8; }}
        .section {{
            padding: 30px;
        }}
        .section h2 {{
            margin: 0 0 20px 0;
            color: #495057;
            font-size: 1.5em;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .score-excellent {{ color: #28a745; font-weight: bold; }}
        .score-good {{ color: #17a2b8; font-weight: bold; }}
        .score-poor {{ color: #dc3545; font-weight: bold; }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #e9ecef;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 AI Model Test Results</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <div class="summary-grid">
                <div class="summary-item">
                    <h3>Total Tests</h3>
                    <div class="value info">{total_tests}</div>
                </div>
                <div class="summary-item">
                    <h3>Tests Passed</h3>
                    <div class="value success">{passed_tests}</div>
                </div>
                <div class="summary-item">
                    <h3>Tests Failed</h3>
                    <div class="value danger">{failed_tests}</div>
                </div>
                <div class="summary-item">
                    <h3>Success Rate</h3>
                    <div class="value {'success' if success_rate >= 80 else 'warning' if success_rate >= 50 else 'danger'}">{success_rate:.1f}%</div>
                </div>
                <div class="summary-item">
                    <h3>Models Tested</h3>
                    <div class="value info">{len(all_models)}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Model Comparison Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Score</th>
                        <th>Pass Rate</th>
                        <th>Avg Distance</th>
                        <th>Tests</th>
                    </tr>
                </thead>
                <tbody>"""
            
            # Add model comparison rows
            for i, (model, stats) in enumerate(sorted_models):
                score_class = "score-excellent" if stats['score'] >= 80 else "score-good" if stats['score'] >= 50 else "score-poor"
                html_content += f"""
                    <tr>
                        <td><strong>{model}</strong></td>
                        <td class="{score_class}">{stats['score']:.1f}</td>
                        <td>{stats['pass_rate']:.1f}%</td>
                        <td>{stats['avg_distance']:.1f}</td>
                        <td>{stats['passed']}/{stats['total']}</td>
                    </tr>"""
            
            html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>📋 Test-Model Matrix</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test</th>"""
            
            # Add model headers
            for model in all_models:
                html_content += f"<th>{model}</th>"
            
            html_content += """
                    </tr>
                </thead>
                <tbody>"""
            
            # Add test-model matrix rows
            for test_name, test_results in results.items():
                html_content += f"""
                    <tr>
                        <td><strong>{test_name}</strong></td>"""
                
                for model in all_models:
                    model_test_results = [r for r in test_results if r.model_name == model]
                    if model_test_results:
                        passed = sum(1 for r in model_test_results if r.passed)
                        total = len(model_test_results)
                        percentage = (passed / total * 100) if total > 0 else 0
                        color_class = "success" if percentage == 100 else "warning" if percentage >= 50 else "danger"
                        html_content += f'<td class="{color_class}">{passed}/{total} ({percentage:.0f}%)</td>'
                    else:
                        html_content += '<td>-</td>'
                
                html_content += "</tr>"
            
            html_content += f"""
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>⚡ Test-Model Latency Table (P95)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test</th>"""
            
            # Add model headers for latency table
            for model in all_models:
                html_content += f"<th>{model}</th>"
            
            html_content += """
                    </tr>
                </thead>
                <tbody>"""
            
            # Calculate and add latency data
            for test_name in all_tests:
                html_content += f"""
                    <tr>
                        <td><strong>{test_name}</strong></td>"""
                
                for model_name in all_models:
                    model_test_results = [r for r in results[test_name] if r.model_name == model_name and r.latency_ms is not None]
                    if model_test_results:
                        latencies = [r.latency_ms for r in model_test_results]
                        latencies.sort()
                        
                        # Calculate P95
                        p95_index = int(len(latencies) * 0.95)
                        if p95_index >= len(latencies):
                            p95_index = len(latencies) - 1
                        p95_latency = latencies[p95_index] if latencies else 0
                        
                        # Color coding for latency
                        if p95_latency < 1000:
                            color_class = "success"
                        elif p95_latency < 2000:
                            color_class = "warning"
                        else:
                            color_class = "danger"
                        
                        html_content += f'<td class="{color_class}">{p95_latency:.0f}ms</td>'
                    else:
                        html_content += '<td>-</td>'
                
                html_content += "</tr>"
            
            html_content += f"""
                </tbody>
            </table>
            <p><small>Values shown are P95 (95th percentile) latencies. Green: &lt;1000ms, Yellow: 1000-2000ms, Red: &gt;2000ms</small></p>
        </div>
        
        <div class="footer">
            <p>Generated by AI Model Testbed • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>"""
            
            # Write HTML file
            with open(self.html_file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"\n{Fore.GREEN}📄 Results exported to: {self.html_file_path}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Warning: Failed to export results to HTML: {e}{Style.RESET_ALL}")
    
    def _get_provider_semaphore(self, model_name: str) -> Semaphore:
        """Get the appropriate rate limiter for a model."""
        if model_name not in self.models_config.models:
            return self.rate_limiters['echo']  # Default fallback
        
        provider = self.models_config.models[model_name].provider
        return self.rate_limiters.get(provider, self.rate_limiters['echo'])
    
    def _validate_all_model_api_keys(self) -> None:
        """Validate API keys for all models before starting execution."""
        for model_run in self.test_run_config.models:
            model_name = model_run.name
            if model_name in self.models_config.models:
                model_config = self.models_config.models[model_name]
                # Create a temporary connector to validate API key
                try:
                    # Only validate API keys, don't actually make network calls
                    # Check if it's a local provider first
                    if self._is_local_provider(model_config.endpoint, model_name):
                        continue  # Skip validation for local providers
                    
                    # For remote providers, just check if API key exists and is not dummy
                    if not model_config.api_key or model_config.api_key.strip() == "" or model_config.api_key in ["dummy", "test-key", "mock-key", "${OPENAI_API_KEY}", "${ANTHROPIC_API_KEY}"]:
                        provider_name = self._get_provider_name(model_config.endpoint)
                        env_var_name = self._get_env_var_name(model_config.endpoint)
                        print(f"\n{Fore.RED}🚫 API key is missing or invalid for {provider_name} model '{model_name}'{Style.RESET_ALL}")
                        print(f"   Please set the {env_var_name} environment variable")
                        print(f"   Example: $env:{env_var_name}=\"your-api-key-here\"")
                        print(f"{Fore.YELLOW}💡 Tip: Set your API key and try again{Style.RESET_ALL}")
                        print(f"{Fore.RED}❌ Execution stopped due to missing API key{Style.RESET_ALL}")
                        import sys
                        sys.exit(1)
                        
                except Exception as e:
                    # If there's any other error, just continue - let the actual test handle it
                    continue
    
    def _is_local_provider(self, endpoint: str, model_name: str) -> bool:
        """Check if this is a local provider that doesn't need API keys."""
        return (
            "mock://" in endpoint or 
            endpoint.startswith("mock://") or
            model_name.startswith("echo-") or
            model_name.startswith("mock-")
        )

    def _get_provider_name(self, endpoint: str) -> str:
        """Get human-readable provider name."""
        if "openai" in endpoint:
            return "OpenAI"
        elif "anthropic" in endpoint:
            return "Anthropic"
        else:
            return "API"

    def _get_env_var_name(self, endpoint: str) -> str:
        """Get the environment variable name for this provider."""
        if "openai" in endpoint:
            return "OPENAI_API_KEY"
        elif "anthropic" in endpoint:
            return "ANTHROPIC_API_KEY"
        else:
            return "API_KEY"
    
    def run_single_test(self, test_name: str, model_name: str, run_number: int = 1, total_runs: int = 1) -> TestResult:
        """Run a single test against a specific model with rate limiting."""
        if test_name not in self.tests_config.tests:
            result = TestResult(
                test_name=test_name,
                model_name=model_name,
                passed=False,
                expected="",
                actual="",
                run_number=run_number,
                error=f"Test '{test_name}' not found in configuration",
                distance=None,
                latency_ms=0.0
            )
            # Log failed tests
            self._log_failed_test(result, total_runs)
            return result
        
        test_config = self.tests_config.tests[test_name]
        
        # Check if model exists in models config
        if model_name not in self.models_config.models:
            result = TestResult(
                test_name=test_name,
                model_name=model_name,
                passed=False,
                expected=test_config.expected_output,
                actual="",
                run_number=run_number,
                error=f"Model '{model_name}' not found in models configuration",
                distance=None,
                latency_ms=0.0
            )
            # Log failed tests
            self._log_failed_test(result, total_runs)
            return result
        
        # Acquire rate limiter for this model's provider
        semaphore = self._get_provider_semaphore(model_name)
        
        with semaphore:
            start_time = time.time()
            
            try:
                # Create connector for the model
                connector = create_connector(model_name, self.models_config)
                
                # Generate response and measure latency
                result = connector.generate(test_config.prompt)
                end_time = time.time()
                
                # Use first-byte latency if available, otherwise fall back to full response time
                if result.first_byte_latency_ms is not None:
                    latency_ms = result.first_byte_latency_ms
                else:
                    latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
                
                actual_output = result.text
                
                # Check if test passes
                if test_config.exact_match:
                    # Normalize whitespace for comparison
                    expected_normalized = normalize_whitespace(test_config.expected_output)
                    actual_normalized = normalize_whitespace(actual_output)
                    passed = expected_normalized == actual_normalized
                    # Calculate lexicographical distance for exact match tests
                    distance = levenshtein_distance(expected_normalized, actual_normalized)
                else:
                    passed = test_config.expected_output.lower() in actual_output.lower()
                    distance = None  # No distance calculation for substring matches
                
                result = TestResult(
                    test_name=test_name,
                    model_name=model_name,
                    passed=passed,
                    expected=test_config.expected_output,
                    actual=actual_output,
                    run_number=run_number,
                    error=None,
                    distance=distance,
                    latency_ms=latency_ms
                )
                # Log failed tests
                self._log_failed_test(result, total_runs)
                return result
                
            except ValueError as e:
                # Handle other validation errors
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                result = TestResult(
                    test_name=test_name,
                    model_name=model_name,
                    passed=False,
                    expected=test_config.expected_output,
                    actual="",
                    run_number=run_number,
                    error=str(e),
                    distance=None,
                    latency_ms=latency_ms
                )
                # Log failed tests
                self._log_failed_test(result, total_runs)
                return result
            except Exception as e:
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                result = TestResult(
                    test_name=test_name,
                    model_name=model_name,
                    passed=False,
                    expected=test_config.expected_output,
                    actual="",
                    run_number=run_number,
                    error=str(e),
                    distance=None,
                    latency_ms=latency_ms
                )
                # Log failed tests
                self._log_failed_test(result, total_runs)
                return result
    
    def run_test(self, test_name: str) -> List[TestResult]:
        """Run a test against all configured models for that test with parallel model execution."""
        if test_name not in self.tests_config.tests:
            return []
        
        results = []
        
        # Validate API keys for all models before starting parallel execution
        self._validate_all_model_api_keys()
        
        # Create all test tasks for this test (all models × all runs)
        test_tasks = []
        for model_run in self.test_run_config.models:
            model_name = model_run.name
            # Use model-specific runs if explicitly specified, otherwise use runs_per_test
            # Check if runs was explicitly set by looking at the model_run object's __fields_set__
            if hasattr(model_run, '__fields_set__') and 'runs' in model_run.__fields_set__:
                runs = model_run.runs
            else:
                runs = self.test_run_config.runs_per_test
            
            for run_num in range(1, runs + 1):
                test_tasks.append((test_name, model_name, run_num))
        
        # Calculate total runs for this test
        total_runs = len(test_tasks)
        completed_runs = 0
        
        # Execute all model runs for this test in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all model tasks for this test
            future_to_task = {
                executor.submit(self.run_single_test, task_test_name, task_model_name, task_run_num, total_runs): (task_test_name, task_model_name, task_run_num)
                for task_test_name, task_model_name, task_run_num in test_tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                    completed_runs += 1
                    
                    # Calculate and show progress within this test
                    progress_percentage = int((completed_runs / total_runs) * 100)
                    # Clear the line and print progress (ensure we clear any previous content)
                    print(f"\r{' ' * 100}\r{Fore.CYAN}🔄 {test_name}{Style.RESET_ALL} progress: {completed_runs}/{total_runs} runs ({progress_percentage}%)", end="", flush=True)
                    
                except Exception as e:
                    task_test_name, task_model_name, task_run_num = future_to_task[future]
                    # Get expected output from test config for proper error reporting
                    expected_output = ""
                    if task_test_name in self.tests_config.tests:
                        expected_output = self.tests_config.tests[task_test_name].expected_output
                    
                    # Create error result
                    error_result = TestResult(
                        test_name=task_test_name,
                        model_name=task_model_name,
                        passed=False,
                        expected=expected_output,
                        actual="",
                        run_number=task_run_num,
                        error=f"Model execution failed: {str(e)}",
                        distance=None
                    )
                    # Log failed tests
                    self._log_failed_test(error_result, total_runs)
                    results.append(error_result)
                    completed_runs += 1
                    
                    # Calculate and show progress within this test
                    progress_percentage = int((completed_runs / total_runs) * 100)
                    # Clear the line and print progress (ensure we clear any previous content)
                    print(f"\r{' ' * 100}\r{Fore.CYAN}🔄 {test_name}{Style.RESET_ALL} progress: {completed_runs}/{total_runs} runs ({progress_percentage}%)", end="", flush=True)
        
        return results
    
    def run_all_tests(self) -> Dict[str, List[TestResult]]:
        """Run all tests against their configured models with parallel execution."""
        all_results = {}
        
        # Validate API keys for all models before starting execution
        self._validate_all_model_api_keys()
        
        # Get all test names from tests-cases.yaml
        test_names = list(self.tests_config.tests.keys())
        
        # Calculate total number of test runs
        total_runs = len(test_names) * self.test_run_config.runs_per_test * len(self.test_run_config.models)
        
        print(f"\n{Fore.CYAN}🚀 Starting parallel test execution ..{Style.RESET_ALL}")
        print(f"   {Fore.YELLOW}Tests:{Style.RESET_ALL} {len(test_names)}")
        print(f"   {Fore.YELLOW}Total Runs:{Style.RESET_ALL} {total_runs}")
        print(f"   {Fore.YELLOW}Models:{Style.RESET_ALL} {len(self.test_run_config.models)}")
        print(f"   {Fore.YELLOW}Max Workers:{Style.RESET_ALL} {self.max_workers}")
        print(f"   {Fore.YELLOW}Rate Limits:{Style.RESET_ALL} OpenAI: 10, Anthropic: 5, Local: 20")
        print()
        
        # Create a list of all test tasks to run in parallel
        test_tasks = []
        for test_name in test_names:
            test_tasks.append((test_name, self.run_test))
        
        # Execute tests in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all test tasks
            future_to_test = {
                executor.submit(test_func, test_name): test_name 
                for test_name, test_func in test_tasks
            }
            
            # Collect results as they complete
            completed_tests = 0
            total_tests = len(test_names)
            
            for future in as_completed(future_to_test):
                test_name = future_to_test[future]
                try:
                    results = future.result()
                    all_results[test_name] = results
                    completed_tests += 1
                    
                    # Show test completion summary
                    test_runs = len(results)
                    passed = sum(1 for r in results if r.passed)
                    models_tested = len(set(r.model_name for r in results))
                    runs_per_model = test_runs // models_tested if models_tested > 0 and test_runs > 0 else 0
                    
                    # Clear the progress line and print test completion
                    print(f"\r{' ' * 80}\r{Fore.GREEN}✅{Style.RESET_ALL} {Fore.YELLOW}{test_name}{Style.RESET_ALL} completed: {passed}/{test_runs} passed ({models_tested} models × {runs_per_model} runs) ({completed_tests}/{total_tests} tests)")
                    print(f"100%")
                    
                except Exception as e:
                    completed_tests += 1
                    print(f"\r{' ' * 80}\r{Fore.RED}❌{Style.RESET_ALL} {Fore.YELLOW}{test_name}{Style.RESET_ALL} failed: {str(e)}")
                    print(f"100%")
                    all_results[test_name] = []
        
        print(f"\n{Fore.GREEN}🎉 All tests completed!{Style.RESET_ALL}")
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
        
        # Sort models by average distance (ascending - lower distance is better)
        sorted_models = sorted(
            model_stats.items(), 
            key=lambda x: x[1]['avg_distance']
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
            elif score >= 50:
                score_color = Fore.YELLOW
            else:
                score_color = Fore.RED
            
            # Distance color coding
            if avg_distance <= 5:
                distance_color = Fore.GREEN
            elif avg_distance <= 20:
                distance_color = Fore.YELLOW
            else:
                distance_color = Fore.RED
            
            print(f"  {Fore.BLUE}{model_name:<13}{Style.RESET_ALL} {score_color}{score:>6.1f}{Style.RESET_ALL} {overall_pass_rate:>6.1f}% {distance_color}{avg_distance:>8.1f}{Style.RESET_ALL} {stats['passed_tests']:>3}/{stats['total_tests']:<3}")
        
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
        
        # Get all test names and sort by overall pass rate (descending - higher pass rate is better)
        test_names = list(results.keys())
        
        # Calculate pass rate for each test across all models
        test_pass_rates = []
        for test_name in test_names:
            test_results = results[test_name]
            total_passed = sum(1 for r in test_results if r.passed)
            total_tests = len(test_results)
            pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
            test_pass_rates.append((test_name, pass_rate))
        
        # Sort by pass rate (descending)
        test_pass_rates.sort(key=lambda x: x[1], reverse=True)
        test_names = [name for name, _ in test_pass_rates]
        
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
        
        # Test-Model latency table with P95 calculations
        self._print_test_model_latency_table(results)
        
        # Export results to HTML
        self._export_results_to_html(results)
    
    def _print_test_model_latency_table(self, results: Dict[str, List[TestResult]]) -> None:
        """Print TEST-MODEL-LATENCY table with P95 latency calculations."""
        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}TEST-MODEL-LATENCY TABLE{Style.RESET_ALL}")
        print("=" * 80)
        
        # Get all unique models and tests
        all_models = sorted(set(result.model_name for test_results in results.values() for result in test_results))
        all_tests = sorted(results.keys())
        
        if not all_models or not all_tests:
            print("No latency data available.")
            return
        
        # Calculate latency statistics for each test-model combination
        latency_data = {}
        for test_name in all_tests:
            latency_data[test_name] = {}
            for model_name in all_models:
                model_test_results = [r for r in results[test_name] if r.model_name == model_name and r.latency_ms is not None]
                if model_test_results:
                    latencies = [r.latency_ms for r in model_test_results]
                    latencies.sort()
                    
                    # Calculate P95 (95th percentile)
                    p95_index = int(len(latencies) * 0.95)
                    if p95_index >= len(latencies):
                        p95_index = len(latencies) - 1
                    p95_latency = latencies[p95_index] if latencies else 0
                    
                    # Calculate other statistics
                    avg_latency = sum(latencies) / len(latencies) if latencies else 0
                    min_latency = min(latencies) if latencies else 0
                    max_latency = max(latencies) if latencies else 0
                    
                    latency_data[test_name][model_name] = {
                        'p95': p95_latency,
                        'avg': avg_latency,
                        'min': min_latency,
                        'max': max_latency,
                        'count': len(latencies)
                    }
                else:
                    latency_data[test_name][model_name] = {
                        'p95': 0,
                        'avg': 0,
                        'min': 0,
                        'max': 0,
                        'count': 0
                    }
        
        # Calculate dynamic column widths
        test_col_width = max(20, max(len(test) for test in all_tests) + 2)
        model_col_width = max(12, max(len(model) for model in all_models) + 2)
        
        # Print table header
        header = f"{'Test':<{test_col_width}}"
        for model in all_models:
            header += f"{model:<{model_col_width}}"
        print(header)
        print("-" * (test_col_width + model_col_width * len(all_models)))
        
        # Print latency data for each test
        for test_name in all_tests:
            row = f"{test_name:<{test_col_width}}"
            for model_name in all_models:
                data = latency_data[test_name][model_name]
                if data['count'] > 0:
                    # Format P95 latency with color coding
                    p95 = data['p95']
                    if p95 < 1000:
                        color = Fore.GREEN
                    elif p95 < 2000:
                        color = Fore.YELLOW
                    else:
                        color = Fore.RED
                    
                    # Format latency value with proper spacing
                    latency_str = f"{color}{p95:.0f}ms{Style.RESET_ALL}"
                    row += f"{latency_str:<{model_col_width}}"
                else:
                    row += f"{'N/A':<{model_col_width}}"
            print(row)
        
        # Print summary statistics
        print("-" * (test_col_width + model_col_width * len(all_models)))
        print(f"{Fore.YELLOW}Latency Legend:{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}Green: < 1000ms{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}Yellow: 1000-2000ms{Style.RESET_ALL}")
        print(f"  {Fore.RED}Red: > 2000ms{Style.RESET_ALL}")
        print(f"  Values shown are P95 (95th percentile) latencies")
        
