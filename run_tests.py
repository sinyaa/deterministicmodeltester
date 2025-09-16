#!/usr/bin/env python3
"""
CLI script to run model tests defined in YAML configuration files.
"""

import argparse
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_testbed.test_runner import ModelTestRunner

def main():
    parser = argparse.ArgumentParser(description="Run AI model tests")
    parser.add_argument(
        "--test", 
        help="Run specific test by name (default: run all tests)"
    )
    parser.add_argument(
        "--model", 
        help="Run tests against specific model only"
    )
    parser.add_argument(
        "--models-config", 
        default="config/models.yaml",
        help="Path to models configuration file"
    )
    parser.add_argument(
        "--tests-config", 
        default="config/tests-cases.yaml",
        help="Path to tests configuration file"
    )
    parser.add_argument(
        "--test-run-config", 
        default="config/test-run.yaml",
        help="Path to test run configuration file"
    )
    parser.add_argument(
        "--run", 
        help="Path to test run configuration file (alternative to --test-run-config)"
    )
    parser.add_argument(
        "--runs", 
        type=int,
        help="Number of times to run each test (overrides config file)"
    )
    parser.add_argument(
        "--all-models", 
        action="store_true",
        help="Run all tests against all configured models"
    )
    parser.add_argument(
        "--bulk-runs", 
        type=int,
        help="Number of runs for bulk testing (used with --all-models)"
    )
    
    args = parser.parse_args()
    
    # Handle --run argument (takes precedence over --test-run-config)
    test_run_config_path = args.test_run_config
    if args.run:
        test_run_config_path = args.run
    
    try:
        # If using --run, don't pass tests_config as it will be loaded from the config file
        if args.run:
            # When using --run, let the test runner load the tests config from the run config
            runner = ModelTestRunner(args.models_config, "", test_run_config_path)
        else:
            runner = ModelTestRunner(args.models_config, args.tests_config, test_run_config_path)
        
        # Override runs_per_test if specified via command line
        if args.runs is not None:
            runner.test_run_config.runs_per_test = args.runs
            # Update all model run counts
            for model_run in runner.test_run_config.models:
                model_run.runs = args.runs
        
        # Handle --all-models option
        if args.all_models:
            # Generate test runs for all tests against all models
            runs_count = args.bulk_runs or args.runs or runner.test_run_config.runs_per_test
            from ai_testbed.config.loader import ModelRunConfig
            all_models = []
            for model_name in runner.models_config.models:
                all_models.append(ModelRunConfig(name=model_name, runs=runs_count))
            runner.test_run_config.models = all_models
            print(f"Running all {len(runner.tests_config.tests)} tests against all {len(runner.models_config.models)} models with {runs_count} runs each")
            print(f"Total test runs: {len(runner.tests_config.tests) * len(runner.models_config.models) * runs_count}")
        
        if args.test:
            if args.model:
                # Run specific test against specific model
                result = runner.run_single_test(args.test, args.model)
                runner.print_results({args.test: [result]})
            else:
                # Run specific test against all configured models
                results = runner.run_test(args.test)
                runner.print_results({args.test: results})
        else:
            # Run all tests
            results = runner.run_all_tests()
            runner.print_results(results)
            
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
