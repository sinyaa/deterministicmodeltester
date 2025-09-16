🧪 AI Model Testbed is a lightweight Python framework for testing different AI models with a consistent interface.
It provides a clean project structure, configuration-driven model definitions, pluggable connectors, and YAML-based test cases for deterministic evaluation.

## Features

- **Configuration-driven models**: Define models in `config/models.yaml` (provider, endpoint, API key, timeout)
- **Pluggable connectors**: Add new connectors by implementing a BaseConnector
  - MockConnector: returns first 10 characters (mock-gpt, mock-claude) or full text (mock-gpt-2)
  - OpenAIConnector: integrates with OpenAI API for real model testing
  - AnthropicConnector: integrates with Anthropic API for Claude models
  - OpenAIRealtimeWebSocketConnector: integrates with OpenAI Realtime API via WebSocket for real-time models
  - EchoConnector: simple echo connector for local testing
- **Test harness**: Run model evaluations with TestHarness, supporting substring or exact matches
- **YAML-defined test cases**: Add test cases in `config/tests/` (prompt, expected output)
- **Simplified test runs**: Automatically run all tests against all configured models
- **Multiple test runs**: Configure and run each test N times for reliability testing
- **Color-coded output**: Visual pass/fail indicators with real-time progress logging
- **Lexicographical distance**: For exact match tests, shows character distance between expected and actual output
- **Failed test details**: Shows distance, expected, and received output for each failed test
- **Model comparison table**: Side-by-side comparison of all models with distance-based scoring and rankings
- **Robust logging**: Detailed progress indicators, timing information, and error reporting
- **First-byte latency measurement**: Measures true response latency from request start to first data received
- **CI-ready**: GitHub Actions workflow included to run pytest on every push/PR

## Latency Measurement

The testbed measures **first-byte latency** for accurate performance comparison:

- **WebSocket Models (RT)**: Measures time from request start to first `response.output_text.delta` received
- **HTTP Models**: Measures time from request start to complete response received (HTTP responses typically arrive all at once)
- **P95 Calculation**: Shows 95th percentile latency across all test runs for each model
- **Color Coding**: Green (< 1000ms), Yellow (1000-2000ms), Red (> 2000ms)

This provides realistic latency comparisons between different model types and helps identify performance characteristics.

## Project Structure

```
deterministicmodeltester/
├── config/
│   ├── models.yaml                    # Model definitions (providers, endpoints, API keys)
│   ├── evals/                         # Evaluation configurations
│   │   ├── full-run.yaml             # Complete test suite
│   │   ├── small-run.yaml            # Quick tests
│   │   ├── rt-run.yaml               # Real-time models
│   │   ├── languages.yaml            # Language tests
│   │   └── mock-run.yaml             # Mock model tests
│   └── tests/                        # Test case definitions
│       ├── contactcenter-cases.yaml  # Contact center scenarios
│       ├── language-cases.yaml       # Multi-language tests
│       ├── mock-tests-cases.yaml     # Mock model tests
│       └── single-case.yaml          # Single test case
├── src/ai_testbed/                   # Core framework
│   ├── connectors/                   # Model connectors
│   ├── config/                       # Configuration loading
│   ├── harness/                      # Test harness
│   └── test_runner.py                # Main test runner
├── tests/                            # Unit tests
├── run_tests.py                      # CLI entry point
├── load-env.ps1                      # PowerShell environment loader
└── env.example                       # Environment variables template
```

## Environment Setup

Before running tests with real models, you need to set up your API keys:

1. **Create a `.env` file** in the project root:
   ```bash
   # .env
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ANTHROPIC_API_KEY=your_actual_anthropic_api_key_here
   ```

2. **Or set the environment variables** directly:
   ```bash
   export OPENAI_API_KEY=your_actual_openai_api_key_here
   export ANTHROPIC_API_KEY=your_actual_anthropic_api_key_here
   ```

3. **For Windows PowerShell**:
   ```powershell
   $env:OPENAI_API_KEY="your_actual_openai_api_key_here"
   $env:ANTHROPIC_API_KEY="your_actual_anthropic_api_key_here"
   ```

**API Key Requirements:**
- **OpenAI API Key**: Required for all OpenAI models (GPT-3.5, GPT-4, GPT-5, O1, etc.)
- **Anthropic API Key**: Required for all Claude models (Opus, Sonnet, Haiku)
- **Mock Models**: No API key required (use dummy values)

The `.env` file is already included in `.gitignore` to keep your API keys secure.

## Supported Models

The AI Model Testbed supports a wide range of AI models from multiple providers:

### 🤖 **OpenAI Models**

**Legacy Chat Completions API (`/v1/chat/completions`):**
- `gpt-3.5-turbo` - Fast and cost-effective for most tasks
- `gpt-3.5-turbo-16k` - Extended context version of GPT-3.5 Turbo
- `gpt-4` - High-quality reasoning and analysis
- `gpt-4-turbo` - Enhanced version of GPT-4 with improved performance

**Responses API (`/v1/responses`):**
- `gpt-4.1` - Latest GPT-4 model with improved capabilities
- `gpt-4.1-mini` - Smaller, faster version of GPT-4.1
- `gpt-4o` - Multimodal model with vision capabilities
- `gpt-4o-mini` - Compact version of GPT-4o

**Realtime API (`/v1/realtime`):**
- `gpt-4o-mini-realtime-preview` - Real-time preview of GPT-4o mini with WebSocket support
- `gpt-4o-realtime-preview` - Real-time preview of GPT-4o with WebSocket support

**O1 Family:**
- `o1-mini` - Smaller version of the O1 reasoning model
- `o1` - Advanced reasoning model (if available)
- `o3` - Latest reasoning model
- `o3-mini` - Compact version of O3

**GPT-5 Family (availability varies by account):**
- `gpt-5` - Next-generation GPT model
- `gpt-5-mini` - Smaller, faster GPT-5
- `gpt-5-nano` - Most compact GPT-5 variant
- `gpt-5-chat` - Chat-optimized GPT-5

**Open Source Models:**
- `gpt-oss-120b` - 120 billion parameter open source model
- `gpt-oss-20b` - 20 billion parameter open source model

### 🧠 **Anthropic Claude Models**

**Claude 4 Family:**
- `claude-opus-4-1-20250805` - Latest Claude Opus 4.1 (most capable)
- `claude-opus-4-20250514` - Claude Opus 4 (high capability)
- `claude-sonnet-4-20250514` - Claude Sonnet 4 (balanced performance)

**Claude 3 Family:**
- `claude-3-7-sonnet-20250219` - Claude Sonnet 3.7 (enhanced reasoning)
- `claude-3-5-haiku-20241022` - Claude Haiku 3.5 (fast and efficient)
- `claude-3-haiku-20240307` - Claude Haiku 3 (original fast model)

### 🧪 **Local/Test Models**

**Mock Connectors:**
- `mock-gpt` - Returns first 10 characters of input (for testing)
- `mock-gpt-2` - Echoes full input text (for testing)
- `echo-local` - Simple echo connector for local testing

### 🔧 **Model Configuration**

All models are configured in `config/models.yaml` with the following structure:

```yaml
models:
  # OpenAI models
  gpt-4:
    provider: openai
    endpoint: https://api.openai.com/v1/chat/completions
    api_key: ${OPENAI_API_KEY}
    timeout_s: 30

  # Anthropic models
  claude-sonnet-4-20250514:
    provider: anthropic
    endpoint: https://api.anthropic.com/v1/messages
    api_key: ${ANTHROPIC_API_KEY}
    timeout_s: 30

  # Realtime models (WebSocket)
  gpt-4o-mini-realtime-preview:
    provider: openai_realtime_websocket
    endpoint: wss://api.openai.com/v1/realtime
    api_key: ${OPENAI_API_KEY}
    timeout_s: 30

  gpt-4o-realtime-preview:
    provider: openai_realtime_websocket
    endpoint: wss://api.openai.com/v1/realtime
    api_key: ${OPENAI_API_KEY}
    timeout_s: 30

  # Mock models
  mock-gpt:
    provider: mock
    endpoint: mock://local
    api_key: ${MOCK_API_KEY:-dummy}
    timeout_s: 10
```

### 📊 **Model Performance Insights**

Based on testing results, here are some performance characteristics:

**🏆 Top Performers:**
- **Claude Haiku 3.5**: Excellent deterministic behavior and multilingual support
- **Claude Opus 4**: Strong overall performance with good reasoning
- **GPT-4.1**: Reliable and consistent across most test types

**🎯 Specialized Strengths:**
- **Deterministic Text**: Claude models excel at exact text reproduction
- **Multilingual**: Claude Haiku 3.5 shows strong multilingual capabilities
- **Reasoning**: GPT-4 family provides good analytical capabilities
- **Speed**: Claude Haiku and GPT-3.5-turbo offer fast response times

**⚠️ Considerations:**
- Some models may have access restrictions based on your API key
- GPT-5 family models may not be available to all accounts
- Realtime models require WebSocket connections and may have different behavior patterns
- Model availability can change over time

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
# Run all tests with default configuration (3 runs per test per model)
python run_tests.py

# Run with custom number of runs per test
python run_tests.py --runs 5

# Run specific test
python run_tests.py --test deterministic_behavior

# Run specific test against specific model
python run_tests.py --test deterministic_behavior --model mock-gpt

# Run with custom config files
python run_tests.py --models-config config/models.yaml --tests-config config/tests/contactcenter-cases.yaml --test-run-config config/evals/full-run.yaml

# Run with a specific evaluation configuration
python run_tests.py --run config/evals/full-run.yaml

# Run RT models specifically
python run_tests.py --run config/evals/rt-run.yaml

# Run language tests
python run_tests.py --run config/evals/languages.yaml

# Run quick tests
python run_tests.py --run config/evals/small-run.yaml

# Run all tests against all models with high run counts (bulk testing)
python run_tests.py --all-models --bulk-runs 100

# Run all tests against all models with default run count
python run_tests.py --all-models
```

### Configuration

The framework uses three separate configuration files for better separation of concerns:

**Benefits of Simplified Configuration:**
- **Reusable Tests**: Define tests once and run them against multiple models
- **Automatic Test Execution**: All tests from test case files are automatically run against all configured models
- **Simple Model Management**: Just specify models and run counts in evaluation configurations
- **Better Organization**: Clear separation between test logic, model definitions, and execution configurations

**Bulk Testing Capabilities:**
- **All Models Testing**: Run all tests against all configured models with `--all-models`
- **High Volume Testing**: Support for 100+ runs per test for reliability analysis
- **Stress Testing**: Perfect for testing model consistency and performance under load
- **Comprehensive Coverage**: Automatically generates all test-model combinations
- **Distance-Based Scoring**: Models ranked by pass rate minus distance penalty for precise evaluation

#### Models Configuration (`config/models.yaml`)
```yaml
models:
  mock-gpt:
    provider: "mock"
    endpoint: "http://mock.com"
    api_key: "your-api-key"
    timeout_s: 30
  another-model:
    provider: "openai"
    endpoint: "https://api.openai.com/v1/chat/completions"
    api_key: "your-openai-key"
    timeout_s: 30
```

#### Test Cases Configuration (`config/tests/contactcenter-cases.yaml`)
```yaml
# Test definitions - decoupled from models
tests:
  deterministic_behavior:
    name: "Deterministic Behavior Test"
    description: "Tests that the model outputs exactly the expected text"
    prompt: "Your test prompt here"
    expected_output: "Expected response"
    exact_match: true
  
  substring_test:
    name: "Substring Match Test"
    description: "Tests that the model output contains the expected substring"
    prompt: "Say hello to the world"
    expected_output: "hello"
    exact_match: false
```

#### Evaluation Configuration (`config/evals/full-run.yaml`)
```yaml
# Evaluation configuration
tests: config/tests/contactcenter-cases.yaml
runs_per_test: 3  # Default number of runs per test

models:
  - name: "gpt-3.5-turbo"
  - name: "gpt-4"
  - name: "claude-opus-4-1-20250805"
```

#### Custom Evaluation Configurations

You can create multiple evaluation configuration files for different testing scenarios:

```bash
# Run with a specific evaluation configuration
python run_tests.py --run config/evals/small-run.yaml
python run_tests.py --run config/evals/languages.yaml
python run_tests.py --run config/evals/rt-run.yaml
```

**Example evaluation configurations:**

**Quick Test (`config/evals/small-run.yaml`):**
```yaml
tests: config/tests/single-case.yaml
runs_per_test: 1

models:
  - name: "gpt-3.5-turbo"
  - name: "gpt-4"
  - name: "claude-opus-4-1-20250805"
  - name: "gpt-5"
```

**RT Models Test (`config/evals/rt-run.yaml`):**
```yaml
tests: config/tests/contactcenter-cases.yaml
runs_per_test: 1

models:
  - name: "gpt-4o-mini-realtime-preview"
  - name: "gpt-4o-realtime-preview"
```

**Language Tests (`config/evals/languages.yaml`):**
```yaml
tests: config/tests/language-cases.yaml
runs_per_test: 3

models:
  - name: "gpt-3.5-turbo"
  - name: "gpt-5"
```

**Stress Test (`config/evals/full-run.yaml`):**
```yaml
tests: config/tests/contactcenter-cases.yaml
runs_per_test: 10

models:
  - name: "gpt-3.5-turbo"
  - name: "gpt-4"
  - name: "claude-opus-4-1-20250805"
```

### Output Example

When running model tests, you'll see color-coded output with real-time progress, failed test details, and a model comparison table:

```
🚀 Starting test execution...
   Tests: 4
   Total Runs: 16
   Models: 4

📋 Test: deterministic_behavior
------------------------------------------------------------
  → Running deterministic_behavior on mock-gpt... ❌ FAIL (0.00s)
  → Running deterministic_behavior on mock-gpt-2... ✅ PASS (0.05s)
  → Running deterministic_behavior on mock-claude... ❌ FAIL (0.00s)
  → Running deterministic_behavior on gpt-4.1... ✅ PASS (0.65s)
   Test Summary: 2/4 passed

================================================================================
MODEL TEST RESULTS
================================================================================

📊 Test Execution Summary:
  Total Tests: 16
  Tests Passed: 8
  Tests Failed: 8
  Success Rate: 50.0%
  Test Types: 16 exact match, 0 substring

❌ Failed Test Details:
--------------------------------------------------------------------------------
Test: deterministic_behavior | Model: mock-gpt | Run: 1
  Distance: 59
  Expected: This is the test prompt to test deterministic behaviour...
  Received: Return exa

Test: deterministic_behavior | Model: mock-claude | Run: 1
  Distance: 59
  Expected: This is the test prompt to test deterministic behaviour...
  Received: Return exa

================================================================================
MODEL COMPARISON TABLE
================================================================================

Model           Score    Pass%    Avg Dist   Tests
-------------------------------------------------
   gpt-4.1         50.0   50.0%      1.2   2/4
   mock-gpt-2      24.7   25.0%     32.8   1/4
   mock-gpt        33.3    0.0%    167.0   0/4
   mock-claude     33.3    0.0%    167.0   0/4
-------------------------------------------------
```

## Key Features

### 🚀 **Simplified Configuration**
- **Automatic Test Execution**: All tests from test case files are automatically run against all configured models
- **Simple Model Management**: Just specify models and run counts in evaluation configurations
- **No Complex Setup**: No need to manually define test-model combinations
- **Organized Structure**: Clear separation between test cases, model definitions, and evaluation configurations

### 📊 **Comprehensive Testing**
- **Multiple Test Runs**: Configure and run each test N times for reliability testing
- **Real-time Progress**: See test execution progress with color-coded indicators
- **Detailed Failure Analysis**: Shows lexicographical distance, expected, and received output for each failed test

### 🎯 **Model Comparison**
- **Distance-Based Scoring**: Models ranked by pass rate minus distance penalty for precise evaluation
- **Visual Rankings**: Clear ranking system with 🥇🥈🥉 symbols
- **Performance Insights**: Color-coded performance indicators (Excellent/Good/Needs improvement)

### 🔧 **Extensible Architecture**
- **Pluggable Connectors**: Easy to add new model providers
- **Mock Testing**: Built-in mock connectors for testing without API costs
- **Real Model Support**: Ready-to-use connectors for OpenAI and Anthropic APIs
- **Realtime WebSocket Support**: Full WebSocket implementation for OpenAI Realtime API
- **Local Testing**: Echo connector for offline testing and development

### 🌐 **Realtime Models Support**
- **WebSocket Integration**: Native support for OpenAI Realtime API via WebSocket connections
- **Audio Response Handling**: Automatic extraction of text transcripts from audio responses
- **Session Management**: Robust session creation and cleanup for realtime models
- **Error Handling**: Comprehensive error handling and retry mechanisms for WebSocket connections
- **Debug Logging**: Detailed logging for troubleshooting WebSocket communication

### 📈 **Robust Logging**
- **Progress Tracking**: Real-time updates during test execution
- **Timing Information**: Execution time for each test run
- **Error Reporting**: Detailed error information for debugging

## Realtime Models Testing

The framework includes full support for OpenAI's Realtime API models, which use WebSocket connections for real-time communication.

### 🌐 **WebSocket Implementation**

The `OpenAIRealtimeWebSocketConnector` provides:

- **Automatic WebSocket Management**: Handles connection, session creation, and cleanup
- **Audio Response Processing**: Extracts text transcripts from audio responses
- **Session State Management**: Robust session lifecycle management
- **Error Recovery**: Comprehensive error handling and retry mechanisms
- **Debug Logging**: Detailed logging for troubleshooting WebSocket issues

### 🔧 **RT Models Configuration**

RT models are configured with the `openai_realtime_websocket` provider:

```yaml
# config/models.yaml
models:
  gpt-4o-mini-realtime-preview:
    provider: openai_realtime_websocket
    endpoint: wss://api.openai.com/v1/realtime
    api_key: ${OPENAI_API_KEY}
    timeout_s: 30
```

### 🧪 **Testing RT Models**

RT models can be tested using the dedicated configuration:

```bash
# Run RT models with their specific test cases
python run_tests.py --run config/evals/rt-run.yaml

# Run specific RT model
python run_tests.py --run config/evals/rt-run.yaml --model gpt-4o-mini-realtime-preview
```

### 📊 **RT Models Features**

- **Real-time Communication**: Uses WebSocket for low-latency communication
- **Audio Response Support**: Handles audio responses and extracts text transcripts
- **Session-based**: Each test run creates a new WebSocket session
- **Automatic Cleanup**: Sessions are properly closed after each test
- **Debug Output**: Comprehensive logging shows WebSocket message flow

### ⚠️ **RT Models Considerations**

- **WebSocket Dependencies**: Requires `websocket-client` package
- **API Key Requirements**: Uses the same `OPENAI_API_KEY` as other OpenAI models
- **Response Format**: RT models return audio responses that are converted to text
- **Session Management**: Each test creates a new WebSocket session
- **Timeout Handling**: WebSocket connections have their own timeout mechanisms