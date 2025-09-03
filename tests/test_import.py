"""Test ML Courses."""

import ml_courses


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(ml_courses.__name__, str)
