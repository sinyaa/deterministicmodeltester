import pytest
from unittest.mock import Mock, patch
from ai_testbed.test_runner import ModelTestRunner, TestResult
from ai_testbed.config.loader import TestSuiteConfig, TestConfig, AppConfig, ModelConfig


@patch('ai_testbed.test_runner.load_app_config')
@patch('ai_testbed.test_runner.load_test_config')
def test_runner_initialization_with_runs_per_test(mock_load_test, mock_load_app):
    """Test that runner initializes with runs_per_test from config."""
    mock_models_config = AppConfig(
        models={
            "mock-gpt": ModelConfig(
                provider="mock",
                endpoint="http://mock.com",
                api_key="test-key"
            )
        }
    )
    
    mock_tests_config = TestSuiteConfig(
        tests={
            "test1": TestConfig(
                name="Test 1",
                description="Test description",
                prompt="Test prompt",
                expected_output="Expected output",
                exact_match=True,
                models=["mock-gpt"]
            )
        },
        runs_per_test=3
    )
    
    mock_load_app.return_value = mock_models_config
    mock_load_test.return_value = mock_tests_config
    
    runner = ModelTestRunner()
    
    assert runner.tests_config.runs_per_test == 3


@patch('ai_testbed.test_runner.load_app_config')
@patch('ai_testbed.test_runner.load_test_config')
def test_run_single_test_includes_run_number(mock_load_test, mock_load_app):
    """Test that run_single_test includes run_number in TestResult."""
    mock_models_config = AppConfig(
        models={
            "mock-gpt": ModelConfig(
                provider="mock",
                endpoint="http://mock.com",
                api_key="test-key"
            )
        }
    )
    
    mock_tests_config = TestSuiteConfig(
        tests={
            "test1": TestConfig(
                name="Test 1",
                description="Test description",
                prompt="Test prompt",
                expected_output="Expected output",
                exact_match=True,
                models=["mock-gpt"]
            )
        },
        runs_per_test=3
    )
    
    mock_load_app.return_value = mock_models_config
    mock_load_test.return_value = mock_tests_config
    
    runner = ModelTestRunner()
    
    # Mock the connector
    with patch('ai_testbed.test_runner.create_connector') as mock_create:
        mock_connector = Mock()
        mock_connector.generate.return_value = Mock(text="Expected output")
        mock_create.return_value = mock_connector
        
        result = runner.run_single_test("test1", "mock-gpt", run_number=2)
        
        assert result.run_number == 2
        assert result.test_name == "test1"
        assert result.model_name == "mock-gpt"
        assert result.passed is True


@patch('ai_testbed.test_runner.load_app_config')
@patch('ai_testbed.test_runner.load_test_config')
def test_run_test_executes_multiple_runs(mock_load_test, mock_load_app):
    """Test that run_test executes the correct number of runs per test."""
    mock_models_config = AppConfig(
        models={
            "mock-gpt": ModelConfig(
                provider="mock",
                endpoint="http://mock.com",
                api_key="test-key"
            )
        }
    )
    
    mock_tests_config = TestSuiteConfig(
        tests={
            "test1": TestConfig(
                name="Test 1",
                description="Test description",
                prompt="Test prompt",
                expected_output="Expected output",
                exact_match=True,
                models=["mock-gpt"]
            )
        },
        runs_per_test=3
    )
    
    mock_load_app.return_value = mock_models_config
    mock_load_test.return_value = mock_tests_config
    
    runner = ModelTestRunner()
    
    # Mock the connector
    with patch('ai_testbed.test_runner.create_connector') as mock_create:
        mock_connector = Mock()
        mock_connector.generate.return_value = Mock(text="Expected output")
        mock_create.return_value = mock_connector
        
        results = runner.run_test("test1")
        
        # Should have 3 results (runs_per_test = 3)
        assert len(results) == 3
        
        # All results should have different run numbers
        run_numbers = [result.run_number for result in results]
        assert set(run_numbers) == {1, 2, 3}
        
        # All results should be for the same test and model
        for result in results:
            assert result.test_name == "test1"
            assert result.model_name == "mock-gpt"


@patch('ai_testbed.test_runner.load_app_config')
@patch('ai_testbed.test_runner.load_test_config')
def test_run_test_with_different_models(mock_load_test, mock_load_app):
    """Test that run_test works with multiple models."""
    # Create config with multiple models
    multi_model_tests_config = TestSuiteConfig(
        tests={
            "test1": TestConfig(
                name="Test 1",
                description="Test description",
                prompt="Test prompt",
                expected_output="Expected output",
                exact_match=True,
                models=["mock-gpt", "mock-gpt-2"]
            )
        },
        runs_per_test=2
    )
    
    multi_model_models_config = AppConfig(
        models={
            "mock-gpt": ModelConfig(
                provider="mock",
                endpoint="http://mock.com",
                api_key="test-key"
            ),
            "mock-gpt-2": ModelConfig(
                provider="mock",
                endpoint="http://mock2.com",
                api_key="test-key-2"
            )
        }
    )
    
    mock_load_app.return_value = multi_model_models_config
    mock_load_test.return_value = multi_model_tests_config
    
    runner = ModelTestRunner()
    
    # Mock the connector
    with patch('ai_testbed.test_runner.create_connector') as mock_create:
        mock_connector = Mock()
        mock_connector.generate.return_value = Mock(text="Expected output")
        mock_create.return_value = mock_connector
        
        results = runner.run_test("test1")
        
        # Should have 4 results (2 models × 2 runs each)
        assert len(results) == 4
        
        # Check that we have results for both models
        model_names = set(result.model_name for result in results)
        assert model_names == {"mock-gpt", "mock-gpt-2"}
        
        # Check that each model has 2 runs
        for model_name in model_names:
            model_results = [r for r in results if r.model_name == model_name]
            assert len(model_results) == 2
            assert set(r.run_number for r in model_results) == {1, 2}


@patch('ai_testbed.test_runner.load_app_config')
@patch('ai_testbed.test_runner.load_test_config')
def test_run_all_tests_with_multiple_runs(mock_load_test, mock_load_app):
    """Test that run_all_tests works with multiple runs."""
    mock_models_config = AppConfig(
        models={
            "mock-gpt": ModelConfig(
                provider="mock",
                endpoint="http://mock.com",
                api_key="test-key"
            )
        }
    )
    
    mock_tests_config = TestSuiteConfig(
        tests={
            "test1": TestConfig(
                name="Test 1",
                description="Test description",
                prompt="Test prompt",
                expected_output="Expected output",
                exact_match=True,
                models=["mock-gpt"]
            )
        },
        runs_per_test=3
    )
    
    mock_load_app.return_value = mock_models_config
    mock_load_test.return_value = mock_tests_config
    
    runner = ModelTestRunner()
    
    # Mock the connector
    with patch('ai_testbed.test_runner.create_connector') as mock_create:
        mock_connector = Mock()
        mock_connector.generate.return_value = Mock(text="Expected output")
        mock_create.return_value = mock_connector
        
        all_results = runner.run_all_tests()
        
        # Should have results for test1
        assert "test1" in all_results
        assert len(all_results["test1"]) == 3  # runs_per_test = 3


def test_test_result_creation_with_run_number():
    """Test TestResult creation with run_number."""
    result = TestResult(
        test_name="test1",
        model_name="mock-gpt",
        passed=True,
        expected="Expected",
        actual="Actual",
        run_number=5
    )
    
    assert result.run_number == 5
    assert result.test_name == "test1"
    assert result.model_name == "mock-gpt"
    assert result.passed is True


@patch('ai_testbed.test_runner.load_app_config')
@patch('ai_testbed.test_runner.load_test_config')
def test_print_results_with_multiple_runs(mock_load_test, mock_load_app, capsys):
    """Test that print_results displays multiple runs correctly."""
    mock_models_config = AppConfig(
        models={
            "mock-gpt": ModelConfig(
                provider="mock",
                endpoint="http://mock.com",
                api_key="test-key"
            )
        }
    )
    
    mock_tests_config = TestSuiteConfig(
        tests={
            "test1": TestConfig(
                name="Test 1",
                description="Test description",
                prompt="Test prompt",
                expected_output="Expected output",
                exact_match=True,
                models=["mock-gpt"]
            )
        },
        runs_per_test=3
    )
    
    mock_load_app.return_value = mock_models_config
    mock_load_test.return_value = mock_tests_config
    
    runner = ModelTestRunner()
    
    # Create mock results with multiple runs
    results = {
        "test1": [
            TestResult("test1", "mock-gpt", True, "Expected", "Actual", 1),
            TestResult("test1", "mock-gpt", False, "Expected", "Different", 2),
            TestResult("test1", "mock-gpt", True, "Expected", "Actual", 3),
        ]
    }
    
    runner.print_results(results)
    captured = capsys.readouterr()
    
    # Check that output contains run information
    assert "Run 1" in captured.out
    assert "Run 2" in captured.out
    assert "Run 3" in captured.out
    assert "Pass Rate:" in captured.out
    assert "66.7%" in captured.out  # 2 out of 3 passed


@patch('ai_testbed.test_runner.load_app_config')
@patch('ai_testbed.test_runner.load_test_config')
def test_print_results_with_single_run(mock_load_test, mock_load_app, capsys):
    """Test that print_results works correctly with single runs."""
    single_run_config = TestSuiteConfig(
        tests={
            "test1": TestConfig(
                name="Test 1",
                description="Test description",
                prompt="Test prompt",
                expected_output="Expected output",
                exact_match=True,
                models=["mock-gpt"]
            )
        },
        runs_per_test=1
    )
    
    mock_models_config = AppConfig(
        models={
            "mock-gpt": ModelConfig(
                provider="mock",
                endpoint="http://mock.com",
                api_key="test-key"
            )
        }
    )
    
    mock_load_app.return_value = mock_models_config
    mock_load_test.return_value = single_run_config
    
    runner = ModelTestRunner()
    
    # Create mock results with single run
    results = {
        "test1": [
            TestResult("test1", "mock-gpt", True, "Expected", "Actual", 1),
        ]
    }
    
    runner.print_results(results)
    captured = capsys.readouterr()
    
    # Check that output doesn't show run numbers for single runs
    assert "Run 1" not in captured.out
    # Check that per-model pass rate is not shown for single runs (but overall pass rate is)
    # Look for per-model pass rate pattern, not overall pass rate
    assert "Pass Rate: 100.0% (1/1)" not in captured.out  # No per-model pass rate for single runs
    assert "✅ PASS" in captured.out
    assert "Overall Pass Rate:" in captured.out  # Overall pass rate is still shown


@patch('ai_testbed.test_runner.load_app_config')
@patch('ai_testbed.test_runner.load_test_config')
def test_print_results_color_coding(mock_load_test, mock_load_app, capsys):
    """Test that print_results uses color coding correctly."""
    mock_models_config = AppConfig(
        models={
            "mock-gpt": ModelConfig(
                provider="mock",
                endpoint="http://mock.com",
                api_key="test-key"
            )
        }
    )
    
    mock_tests_config = TestSuiteConfig(
        tests={
            "test1": TestConfig(
                name="Test 1",
                description="Test description",
                prompt="Test prompt",
                expected_output="Expected output",
                exact_match=True,
                models=["mock-gpt"]
            )
        },
        runs_per_test=3
    )
    
    mock_load_app.return_value = mock_models_config
    mock_load_test.return_value = mock_tests_config
    
    runner = ModelTestRunner()
    
    # Create mock results with mixed pass/fail
    results = {
        "test1": [
            TestResult("test1", "mock-gpt", True, "Expected", "Actual", 1),
            TestResult("test1", "mock-gpt", False, "Expected", "Different", 2),
        ]
    }
    
    runner.print_results(results)
    captured = capsys.readouterr()
    
    # Check that color codes are present (colorama adds ANSI codes)
    assert "\x1b[" in captured.out  # ANSI color codes
    assert "MODEL TEST RESULTS" in captured.out
    assert "SUMMARY:" in captured.out
    assert "Overall Pass Rate:" in captured.out