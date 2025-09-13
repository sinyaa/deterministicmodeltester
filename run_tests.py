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
        default="config/tests.yaml",
        help="Path to tests configuration file"
    )
    parser.add_argument(
        "--runs", 
        type=int,
        help="Number of times to run each test (overrides config file)"
    )
    
    args = parser.parse_args()
    
    try:
        runner = ModelTestRunner(args.models_config, args.tests_config)
        
        # Override runs_per_test if specified via command line
        if args.runs is not None:
            runner.tests_config.runs_per_test = args.runs
        
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
