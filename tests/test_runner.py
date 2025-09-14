import pytest
from unittest.mock import Mock, patch
from ai_testbed.test_runner import ModelTestRunner, TestResult
from ai_testbed.config.loader import TestSuiteConfig, TestConfig, AppConfig, ModelConfig, TestRunConfig, ModelRunConfig


@patch('ai_testbed.test_runner.load_app_config')
@patch('ai_testbed.test_runner.load_test_config')
@patch('ai_testbed.test_runner.load_test_run_config')
def test_runner_initialization_with_runs_per_test(mock_load_test_run, mock_load_test, mock_load_app):
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
                exact_match=True
            )
        }
    )

    mock_test_run_config = TestRunConfig(
            models=[
                ModelRunConfig(name="mock-gpt", runs=3)
            ],
            runs_per_test=3
        )
    
    mock_load_app.return_value = mock_models_config
    mock_load_test.return_value = mock_tests_config
    mock_load_test_run.return_value = mock_test_run_config
    
    runner = ModelTestRunner("dummy", "dummy", "dummy")
    
    assert runner.test_run_config.runs_per_test == 3


@patch('ai_testbed.test_runner.load_app_config')
@patch('ai_testbed.test_runner.load_test_config')
@patch('ai_testbed.test_runner.load_test_run_config')
def test_run_single_test_includes_run_number(mock_load_test_run, mock_load_test, mock_load_app):
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
    
    runner = ModelTestRunner("dummy", "dummy", "dummy")
    
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
@patch('ai_testbed.test_runner.load_test_run_config')
def test_run_test_executes_multiple_runs(mock_load_test_run, mock_load_test, mock_load_app):
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
                exact_match=True
            )
        }
    )

    mock_test_run_config = TestRunConfig(
            models=[
                ModelRunConfig(name="mock-gpt", runs=3)
            ],
            runs_per_test=3
        )
    
    mock_load_app.return_value = mock_models_config
    mock_load_test.return_value = mock_tests_config
    mock_load_test_run.return_value = mock_test_run_config
    
    runner = ModelTestRunner("dummy", "dummy", "dummy")
    
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
@patch('ai_testbed.test_runner.load_test_run_config')
def test_run_test_with_different_models(mock_load_test_run, mock_load_test, mock_load_app):
    """Test that run_test works with multiple models."""
    # Create config with multiple models
    multi_model_tests_config = TestSuiteConfig(
        tests={
            "test1": TestConfig(
                name="Test 1",
                description="Test description",
                prompt="Test prompt",
                expected_output="Expected output",
                exact_match=True
            )
        }
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

    multi_model_test_run_config = TestRunConfig(
            models=[
                ModelRunConfig(name="mock-gpt", runs=2),
                ModelRunConfig(name="mock-gpt-2", runs=2)
            ],
            runs_per_test=2
        )
    
    mock_load_app.return_value = multi_model_models_config
    mock_load_test.return_value = multi_model_tests_config
    mock_load_test_run.return_value = multi_model_test_run_config
    
    runner = ModelTestRunner("dummy", "dummy", "dummy")
    
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
@patch('ai_testbed.test_runner.load_test_run_config')
def test_run_all_tests_with_multiple_runs(mock_load_test_run, mock_load_test, mock_load_app):
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
                exact_match=True
            )
        }
    )

    mock_test_run_config = TestRunConfig(
            models=[
                ModelRunConfig(name="mock-gpt", runs=3)
            ],
            runs_per_test=3
        )
    
    mock_load_app.return_value = mock_models_config
    mock_load_test.return_value = mock_tests_config
    mock_load_test_run.return_value = mock_test_run_config
    
    runner = ModelTestRunner("dummy", "dummy", "dummy")
    
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


def test_levenshtein_distance_calculation():
    """Test the Levenshtein distance calculation function."""
    from ai_testbed.test_runner import levenshtein_distance
    
    # Test identical strings
    assert levenshtein_distance("hello", "hello") == 0
    
    # Test single character difference
    assert levenshtein_distance("hello", "hallo") == 1
    
    # Test insertion
    assert levenshtein_distance("hello", "hellos") == 1
    
    # Test deletion
    assert levenshtein_distance("hello", "hell") == 1
    
    # Test substitution
    assert levenshtein_distance("hello", "hallo") == 1
    
    # Test multiple differences
    assert levenshtein_distance("kitten", "sitting") == 3
    
    # Test empty strings
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("hello", "") == 5
    assert levenshtein_distance("", "hello") == 5


@patch('ai_testbed.test_runner.load_test_run_config')
@patch('ai_testbed.test_runner.load_app_config')
@patch('ai_testbed.test_runner.load_test_config')
def test_exact_match_test_calculates_distance(mock_load_test, mock_load_app, mock_load_test_run):
    """Test that exact match tests calculate lexicographical distance."""
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
                expected_output="Hello World",
                exact_match=True  # This should trigger distance calculation
            )
        }
    )

    mock_test_run_config = TestRunConfig(
        models=[
            ModelRunConfig(name="mock-gpt", runs=1)
        ],
        runs_per_test=1
    )
    
    mock_load_app.return_value = mock_models_config
    mock_load_test.return_value = mock_tests_config
    mock_load_test_run.return_value = mock_test_run_config
    
    runner = ModelTestRunner("dummy", "dummy", "dummy")
    
    # Mock the connector to return a different output
    with patch('ai_testbed.test_runner.create_connector') as mock_create:
        mock_connector = Mock()
        mock_connector.generate.return_value = Mock(text="Hello Word")  # 1 character difference
        mock_create.return_value = mock_connector
        
        result = runner.run_single_test("test1", "mock-gpt", run_number=1)
        
        assert result.passed is False
        assert result.distance == 1  # "Hello World" vs "Hello Word" = 1 character difference
        assert result.expected == "Hello World"
        assert result.actual == "Hello Word"


@patch('ai_testbed.test_runner.load_test_run_config')
@patch('ai_testbed.test_runner.load_app_config')
@patch('ai_testbed.test_runner.load_test_config')
def test_exact_match_test_passing_case(mock_load_test, mock_load_app, mock_load_test_run):
    """Test that exact match tests work correctly when they pass."""
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
                    expected_output="Hello World",
                    exact_match=True  # This should trigger distance calculation
                )
            }
    )

    mock_test_run_config = TestRunConfig(
        models=[
            ModelRunConfig(name="mock-gpt", runs=1)
        ],
        runs_per_test=1
    )

    mock_load_app.return_value = mock_models_config
    mock_load_test.return_value = mock_tests_config
    mock_load_test_run.return_value = mock_test_run_config

    runner = ModelTestRunner("dummy", "dummy", "dummy")
    
    # Mock the connector to return the exact expected output
    with patch('ai_testbed.test_runner.create_connector') as mock_create:
        mock_connector = Mock()
        mock_connector.generate.return_value = Mock(text="Hello World")  # Exact match
        mock_create.return_value = mock_connector
        
        result = runner.run_single_test("test1", "mock-gpt", run_number=1)
        
        assert result.passed is True
        assert result.distance == 0  # "Hello World" vs "Hello World" = 0 character difference
        assert result.expected == "Hello World"
        assert result.actual == "Hello World"
    assert result.error is None


@patch('ai_testbed.test_runner.load_test_run_config')
@patch('ai_testbed.test_runner.load_app_config')
@patch('ai_testbed.test_runner.load_test_config')
def test_substring_match_test_no_distance(mock_load_test, mock_load_app, mock_load_test_run):
    """Test that substring match tests don't calculate distance."""
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
                    expected_output="Hello",
                    exact_match=False  # This should NOT trigger distance calculation
                )
            }
    )

    mock_test_run_config = TestRunConfig(
        models=[
            ModelRunConfig(name="mock-gpt", runs=1)
        ],
        runs_per_test=1
    )

    mock_load_app.return_value = mock_models_config
    mock_load_test.return_value = mock_tests_config
    mock_load_test_run.return_value = mock_test_run_config

    runner = ModelTestRunner("dummy", "dummy", "dummy")
    
    # Mock the connector to return a different output
    with patch('ai_testbed.test_runner.create_connector') as mock_create:
        mock_connector = Mock()
        mock_connector.generate.return_value = Mock(text="Hello World")  # Contains "Hello"
        mock_create.return_value = mock_connector
        
        result = runner.run_single_test("test1", "mock-gpt", run_number=1)
        
        assert result.passed is True
        assert result.distance is None  # No distance calculation for substring matches
        assert result.expected == "Hello"
        assert result.actual == "Hello World"


@patch('ai_testbed.test_runner.load_test_run_config')
@patch('ai_testbed.test_runner.load_app_config')
@patch('ai_testbed.test_runner.load_test_config')
def test_print_results_shows_distance(mock_load_test, mock_load_app, mock_load_test_run, capsys):
    """Test that print_results displays distance in comparison table for exact match tests."""
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
                    expected_output="Hello World",
                    exact_match=True
                )
            }
    )

    mock_test_run_config = TestRunConfig(
        models=[
            ModelRunConfig(name="mock-gpt", runs=1)
        ],
        runs_per_test=1
    )

    mock_load_app.return_value = mock_models_config
    mock_load_test.return_value = mock_tests_config
    mock_load_test_run.return_value = mock_test_run_config

    runner = ModelTestRunner("dummy", "dummy", "dummy")

    # Create mock results with distance
    results = {
        "test1": [
            TestResult("test1", "mock-gpt", False, "Hello World", "Hello Word", 1, None, 1),
        ]
    }

    runner.print_results(results)
    captured = capsys.readouterr()

    # Check that distance is shown in comparison table
    assert "Avg Dist" in captured.out
    assert "1.0" in captured.out  # Distance should be 1.0


@patch('ai_testbed.test_runner.load_test_run_config')
@patch('ai_testbed.test_runner.load_test_config')
@patch('ai_testbed.test_runner.load_app_config')
def test_print_results_with_multiple_runs(mock_load_app, mock_load_test, mock_load_test_run, capsys):
    """Test that print_results displays multiple runs correctly in summary."""
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
                exact_match=True
            )
        }
    )

    mock_test_run_config = TestRunConfig(
            models=[
                ModelRunConfig(name="mock-gpt", runs=3)
            ],
            runs_per_test=3
        )

    mock_load_app.return_value = mock_models_config
    mock_load_test.return_value = mock_tests_config
    mock_load_test_run.return_value = mock_test_run_config

    runner = ModelTestRunner("dummy", "dummy", "dummy")

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

    # Check that output contains test execution summary
    assert "Total Tests: 3" in captured.out
    assert "Tests Passed: 2" in captured.out
    assert "Tests Failed: 1" in captured.out


@patch('ai_testbed.test_runner.load_test_run_config')
@patch('ai_testbed.test_runner.load_test_config')
@patch('ai_testbed.test_runner.load_app_config')
def test_print_results_with_single_run(mock_load_app, mock_load_test, mock_load_test_run, capsys):
    """Test that print_results works correctly with single runs."""
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
                exact_match=True
            )
        }
    )

    mock_test_run_config = TestRunConfig(
            models=[
                ModelRunConfig(name="mock-gpt", runs=1)
            ],
            runs_per_test=1
        )

    mock_load_app.return_value = mock_models_config
    mock_load_test.return_value = mock_tests_config
    mock_load_test_run.return_value = mock_test_run_config

    runner = ModelTestRunner("dummy", "dummy", "dummy")

    # Create mock results with single run
    results = {
        "test1": [
            TestResult("test1", "mock-gpt", True, "Expected", "Actual", 1),
        ]
    }

    runner.print_results(results)
    captured = capsys.readouterr()

    # Check that output shows test execution summary
    assert "Total Tests: 1" in captured.out
    assert "Tests Passed: 1" in captured.out
    assert "Tests Failed: 0" in captured.out
    assert "Overall Pass Rate: 100.0%" in captured.out


@patch('ai_testbed.test_runner.load_test_run_config')
@patch('ai_testbed.test_runner.load_test_config')
@patch('ai_testbed.test_runner.load_app_config')
def test_print_results_color_coding(mock_load_app, mock_load_test, mock_load_test_run, capsys):
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
                exact_match=True
            )
        }
    )

    mock_test_run_config = TestRunConfig(
            models=[
                ModelRunConfig(name="mock-gpt", runs=2)
            ],
            runs_per_test=2
        )

    mock_load_app.return_value = mock_models_config
    mock_load_test.return_value = mock_tests_config
    mock_load_test_run.return_value = mock_test_run_config

    runner = ModelTestRunner("dummy", "dummy", "dummy")

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
    assert "FINAL SUMMARY" in captured.out
    assert "Overall Pass Rate:" in captured.out