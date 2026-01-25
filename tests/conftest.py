"""Conftest for pytest configuration."""

import pytest


def pytest_addoption(parser):
    """Add --slow option to run slow integration tests."""
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow integration tests"
    )


def pytest_configure(config):
    """Register slow marker."""
    config.addinivalue_line("markers", "slow: marks test as slow (skip by default)")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --slow is passed."""
    if config.getoption("--slow"):
        # --slow given, run all tests
        return
    
    skip_slow = pytest.mark.skip(reason="need --slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
