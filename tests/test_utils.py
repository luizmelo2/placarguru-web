import pytest
from unittest.mock import MagicMock
import utils

def test_fetch_release_file_is_secure(mocker):
    """
    Tests that fetch_release_file calls requests.get securely.
    This test should FAIL with the vulnerable code and PASS with the fixed code.
    """
    # Patch the requests.get call
    mock_get = mocker.patch('utils.requests.get')

    # Create a mock response object that returns serializable values
    # because the function is decorated with @st.cache_data
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake excel data"  # bytes are serializable
    # side_effect allows us to return different values on subsequent calls to the same mock
    mock_response.headers.get.side_effect = ["some_etag", "some_last_mod"]
    mock_get.return_value = mock_response

    # Call the function
    utils.fetch_release_file("some_url")

    # This assertion will fail because the vulnerable code includes `verify=False`
    mock_get.assert_called_with("some_url", timeout=60)

    # Clear the cache to avoid side effects between tests
    utils.fetch_release_file.clear()
