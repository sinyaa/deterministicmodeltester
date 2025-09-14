from __future__ import annotations
from pydantic import BaseModel
from typing import Dict, Optional, List
import os, yaml

class ModelConfig(BaseModel):
    provider: str
    endpoint: str
    api_key: str
    timeout_s: int = 30

class TestConfig(BaseModel):
    name: str
    description: str
    prompt: str
    expected_output: str
    exact_match: bool = True

class AppConfig(BaseModel):
    models: Dict[str, ModelConfig]

class TestSuiteConfig(BaseModel):
    tests: Dict[str, TestConfig]

class ModelRunConfig(BaseModel):
    name: str
    runs: int = 1  # Default to 1 run if not specified

class TestRunConfig(BaseModel):
    models: List[ModelRunConfig]  # List of models to test with their run counts
    runs_per_test: int = 1  # Default number of runs per test

def _env_expand(value: str) -> str:
    # Support ${VAR:-default} and ${VAR} interpolation from the shell-ish syntax in YAML
    import re
    def repl(m):
        key = m.group("key")
        default = m.group("default")
        return os.getenv(key, default if default is not None else m.group(0))
    return re.sub(r"\$\{(?P<key>[A-Z0-9_]+)(?::-(?P<default>[^}]*))?\}", repl, value)

def load_app_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Expand env vars only for str leaf nodes
    def walk(obj):
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(x) for x in obj]
        if isinstance(obj, str):
            return _env_expand(obj)
        return obj
    expanded = walk(data)
    return AppConfig(**expanded)

def load_test_config(path: str) -> TestSuiteConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Expand env vars only for str leaf nodes
    def walk(obj):
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(x) for x in obj]
        if isinstance(obj, str):
            return _env_expand(obj)
        return obj
    expanded = walk(data)
    return TestSuiteConfig(**expanded)

def load_test_run_config(path: str) -> TestRunConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Expand env vars only for str leaf nodes
    def walk(obj):
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(x) for x in obj]
        if isinstance(obj, str):
            return _env_expand(obj)
        return obj
    expanded = walk(data)
    return TestRunConfig(**expanded)
