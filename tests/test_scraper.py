import pytest
import requests
from unittest.mock import patch, MagicMock, mock_open

from src.data_acquisition.scraper import SuttaScraper

# --- Test Data and Fixtures ---

@pytest.fixture
def mock_config():
    """Provides a mock configuration for the scraper."""
    return {
        'dhammatalks': {
            'master_url': "https://www.dhammatalks.org/suttas/index_mobile.html",
            'base_url': "https://www.dhammatalks.org",
            'books_of_interest': ["MN", "SN"],
            'avoid_in_url': ["app", "intro"],
        }
    }

@pytest.fixture
def scraper_instance(mock_config):
    """Provides an instance of the SuttaScraper with a mock config."""
    return SuttaScraper(mock_config)

# Mock HTML for the main index page
MOCK_INDEX_HTML = """
<html><body>
    <a href="/suttas/MN/MN1.html">MN 1</a>
    <a href="/suttas/DN/DN1.html">DN 1 (Ignored Book)</a>
    <a href="/suttas/SN/SN1.1.html">SN 1.1</a>
    <a href="/suttas/MN/MN2.html">MN 2</a>
    <a href="/suttas/MN/MN_intro.html">MN Intro (Ignored)</a>
    <a href="/suttas/SN/SN_app.html">SN Appendix (Ignored)</a>
</body></html>
"""

# Mock HTML for a single sutta page
MOCK_SUTTA_HTML = """
<html><body>
<div id="sutta">
    <h1>MN 1: The Root of All Things</h1>
    <p class="seealso">See also: DN 15</p>
    I have heard that on one occasion...
    * * *
    This is the body of the sutta. It contains many teachings.
    <div class="note">This is a note.</div>
</div>
</body></html>
"""

# --- Tests for SuttaScraper Class ---

def test_scraper_initialization(scraper_instance, mock_config):
    """Test that the scraper initializes correctly with the config."""
    assert scraper_instance.master_url == mock_config['dhammatalks']['master_url']
    assert scraper_instance.base_url == mock_config['dhammatalks']['base_url']
    assert scraper_instance.books == ["MN", "SN"]
    assert scraper_instance.avoid == ["app", "intro"]

@patch('src.data_acquisition.scraper.requests.get')
def test_get_sutta_links_success(mock_requests_get, scraper_instance):
    """Test successful parsing of the main index page."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.text = MOCK_INDEX_HTML
    mock_response.raise_for_status.return_value = None
    mock_requests_get.return_value = mock_response

    # Run the method
    links = scraper_instance.get_sutta_links()

    # Assertions
    assert len(links) == 3
    assert links[0]['book'] == 'MN'
    assert links[0]['url'] == "https://www.dhammatalks.org/suttas/MN/MN1.html"
    assert links[1]['url'] == "https://www.dhammatalks.org/suttas/MN/MN2.html"
    assert links[2]['book'] == 'SN'
    assert links[2]['url'] == "https://www.dhammatalks.org/suttas/SN/SN1.1.html"
    mock_requests_get.assert_called_once_with(scraper_instance.master_url)

@patch('src.data_acquisition.scraper.requests.get')
def test_get_sutta_links_request_exception(mock_requests_get, scraper_instance):
    """Test that an empty list is returned on a request exception."""
    mock_requests_get.side_effect = requests.RequestException("Connection failed")
    
    links = scraper_instance.get_sutta_links()
    
    assert links == []

def test_parse_sutta_page(scraper_instance):
    """Test the internal parsing of a single sutta's HTML."""
    parsed_data = scraper_instance._parse_sutta_page(MOCK_SUTTA_HTML)

    assert parsed_data is not None
    assert parsed_data['title'] == "MN 1: The Root of All Things"
    assert parsed_data['introduction'] == "I have heard that on one occasion..."
    assert parsed_data['body'] == "This is the body of the sutta. It contains many teachings."

def test_parse_sutta_page_no_sutta_div(scraper_instance):
    """Test parsing HTML that is missing the main #sutta div."""
    html = "<html><body><p>No sutta here.</p></body></html>"
    parsed_data = scraper_instance._parse_sutta_page(html)
    assert parsed_data is None

@patch('src.data_acquisition.scraper.os.makedirs')
@patch('src.data_acquisition.scraper.jsonlines.open', new_callable=mock_open)
@patch('src.data_acquisition.scraper.requests.get')
def test_run_pipeline(mock_requests_get, mock_jsonlines_open, mock_makedirs, scraper_instance, tmp_path):
    """Test the main `run` method orchestrates the pipeline correctly."""
    output_file = tmp_path / "suttas.jsonl"

    # 1. Mock the response for get_sutta_links
    mock_index_response = MagicMock()
    mock_index_response.text = MOCK_INDEX_HTML
    mock_index_response.raise_for_status.return_value = None

    # 2. Mock the response for the individual sutta pages
    mock_sutta_response = MagicMock()
    mock_sutta_response.text = MOCK_SUTTA_HTML
    mock_sutta_response.raise_for_status.return_value = None

    # The first call to requests.get is for the index, the rest are for suttas
    mock_requests_get.side_effect = [
        mock_index_response, 
        mock_sutta_response, 
        mock_sutta_response, 
        mock_sutta_response
    ]

    # Run the pipeline
    scraper_instance.run(str(output_file))

    # Assertions
    mock_makedirs.assert_called_once_with(str(tmp_path))
    mock_jsonlines_open.assert_called_once_with(str(output_file), mode='w')
    
    # Check that writer.write was called 3 times (for the 3 valid links)
    handle = mock_jsonlines_open()
    assert handle.write.call_count == 3

    # Check the content of the first call to write
    first_call_args = handle.write.call_args[0][0]
    assert first_call_args['sutta_id'] == 1
    assert first_call_args['url'] == "https://www.dhammatalks.org/suttas/MN/MN1.html"
    assert first_call_args['title'] == "MN 1: The Root of All Things"