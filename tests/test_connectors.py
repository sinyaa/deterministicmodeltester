from ai_testbed.config.loader import load_app_config
from ai_testbed.connectors.registry import create_connector

def test_mock_connector_roundtrip():
    cfg = load_app_config("config/models.yaml")
    conn = create_connector("mock-gpt", cfg)
    out = conn.generate("abcd")
    assert out.text.endswith("dcba")
    assert out.model == "mock-gpt"
