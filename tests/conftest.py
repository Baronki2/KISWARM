import pytest

@pytest.fixture
def sample_fixture():
    return "sample data"

@pytest.fixture(scope='session')
def session_test_config():
    # Configure any session-wide settings or resources here
    return {'config_key': 'config_value'}
