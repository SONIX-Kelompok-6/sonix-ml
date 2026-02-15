import pytest
from unittest.mock import MagicMock

@pytest.fixture(autouse=True)
def mock_supabase(mocker):
    """
    Automatically mocks the Supabase client for all tests.
    This prevents real network calls to Supabase during CI.
    """
    # Mock the create_client call in src.database
    mock_client = MagicMock()
    mocker.patch("src.database.create_client", return_value=mock_client)
    
    # Mock the 'supabase' variable itself in src.database
    mocker.patch("src.database.supabase", mock_client)
    
    return mock_client