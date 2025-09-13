🧪 AI Model Testbed is a lightweight Python framework for testing different AI models with a consistent interface.
It provides a clean project structure, configuration-driven model definitions, pluggable connectors, and YAML-based test cases for deterministic evaluation.

Features

Configuration-driven models
Define models in config/models.yaml (provider, endpoint, API key, timeout).

Pluggable connectors
Add new connectors by implementing a BaseConnector. Includes:

MockConnector: returns reversed prompts.

EchoConnector: returns the prompt exactly (deterministic).

Test harness
Run model evaluations with TestHarness, supporting substring or exact matches.

YAML-defined test cases
Add test cases in tests/cases/*.yaml (prompt, expected output).

CI-ready
GitHub Actions workflow included to run pytest on every push/PR.