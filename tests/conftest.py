import pytest


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    """Keep the per-IP rate limiter from leaking between tests."""
    try:
        import security
    except ImportError:
        yield
        return
    security.reset_rate_limit()
    yield
    security.reset_rate_limit()
