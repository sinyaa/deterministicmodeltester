🧪 AI Model Testbed is a lightweight Python framework for testing different AI models with a consistent interface.
It provides a clean project structure, configuration-driven model definitions, pluggable connectors, and YAML-based test cases for deterministic evaluation.

## Features

- **Configuration-driven models**: Define models in `config/models.yaml` (provider, endpoint, API key, timeout)
- **Pluggable connectors**: Add new connectors by implementing a BaseConnector
  - MockConnector: returns reversed prompts
  - EchoConnector: returns the prompt exactly (deterministic)
- **Test harness**: Run model evaluations with TestHarness, supporting substring or exact matches
- **YAML-defined test cases**: Add test cases in `config/tests.yaml` (prompt, expected output)
- **Multiple test runs**: Configure and run each test N times for reliability testing
- **Color-coded output**: Visual pass/fail indicators with pass rate calculations
- **CI-ready**: GitHub Actions workflow included to run pytest on every push/PR

## Quick Start

### Running Unit Tests

To run the test suite and verify the framework is working correctly:

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_runner.py -v

# Run with coverage
python -m pytest tests/ --cov=src/ai_testbed
```

### Running Model Tests

To test actual AI models with your configured test cases:

```bash
# Run all tests with default configuration (3 runs per test)
python run_tests.py

# Run with custom number of runs per test
python run_tests.py --runs 5

# Run specific test
python run_tests.py --test deterministic_behavior

# Run specific test against specific model
python run_tests.py --test deterministic_behavior --model mock-gpt

# Run with custom config files
python run_tests.py --models-config config/models.yaml --tests-config config/tests.yaml
```

### Configuration

#### Models Configuration (`config/models.yaml`)
```yaml
models:
  mock-gpt:
    provider: "mock"
    endpoint: "http://mock.com"
    api_key: "your-api-key"
    timeout_s: 30
```

#### Tests Configuration (`config/tests.yaml`)
```yaml
# Number of times to run each test
runs_per_test: 3

tests:
  deterministic_behavior:
    name: "Deterministic Behavior Test"
    description: "Tests that the model outputs exactly the expected text"
    prompt: "Your test prompt here"
    expected_output: "Expected response"
    exact_match: true
    models:
      - "mock-gpt"
```

### Output Example

When running model tests, you'll see color-coded output like:

```
================================================================================
MODEL TEST RESULTS
================================================================================

Test: deterministic_behavior
----------------------------------------

  Model: mock-gpt
    ✅ PASS (Run 1)
    ❌ FAIL (Run 2)
    ✅ PASS (Run 3)
    Pass Rate: 66.7% (2/3)

================================================================================
SUMMARY: 2/3 tests passed
Overall Pass Rate: 66.7%
================================================================================
```