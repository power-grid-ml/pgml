"""Explicit reference-model fixtures for conformance tests."""

import pytest

from pgml.defaults import use_preset


@pytest.fixture
def opendss_model_defaults():
    """Apply OpenDSS's supported model choices to both comparison arms."""
    with use_preset("opendss"):
        yield
