from __future__ import annotations
from typing import Dict, Type
from .base import BaseConnector
from .mock import MockConnector
from .openai import OpenAIConnector
from .echo import EchoConnector

# Map provider id -> connector class
PROVIDERS: Dict[str, Type[BaseConnector]] = {
    "mock": MockConnector,
    "openai": OpenAIConnector,
    "echo": EchoConnector,
}

def create_connector(model_name: str, cfg) -> BaseConnector:
    if model_name not in cfg.models:
        raise KeyError(f"Unknown model: {model_name}")
    mc = cfg.models[model_name]
    impl = PROVIDERS.get(mc.provider)
    if impl is None:
        raise ValueError(f"No connector registered for provider '{mc.provider}'")
    
    # Create connector with retry parameters
    return impl(
        model_name=model_name, 
        endpoint=mc.endpoint, 
        api_key=mc.api_key, 
        timeout_s=mc.timeout_s,
        max_retries=3,  # Default retry count
        retry_delay=10.0  # Default 10 second base delay
    )
