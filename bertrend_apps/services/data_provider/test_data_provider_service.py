#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import tempfile
from pathlib import Path
from datetime import datetime
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from bertrend_apps.services.data_provider import data_provider_service as svc


# Mock article data for testing
MOCK_ARTICLES = [
    {
        "title": "Test Article 1",
        "summary": "Summary 1",
        "link": "http://example.com/1",
        "url": "http://example.com/1",
        "text": "Article text 1",
        "timestamp": "2024-10-20 10:00:00",
    },
    {
        "title": "Test Article 2",
        "summary": "Summary 2",
        "link": "http://example.com/2",
        "url": "http://example.com/2",
        "text": "Article text 2",
        "timestamp": "2024-10-20 11:00:00",
    },
]


class MockProvider:
    """Mock data provider for testing"""
    
    def get_articles(self, keywords, after, before, max_results, language):
        return MOCK_ARTICLES[:max_results]
    
    def get_articles_batch(self, queries_batch, max_results, language, evaluate_articles_quality, minimum_quality_level):
        # Return mock articles for each query in batch
        return MOCK_ARTICLES * len(queries_batch)
    
    def store_articles(self, articles, save_path):
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                for article in articles:
                    f.write(str(article) + "\n")


@pytest.fixture()
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(svc.app)


@pytest.fixture()
def mock_provider(monkeypatch):
    """Mock all providers with MockProvider"""
    mock_providers = {
        "arxiv": MockProvider,
        "atom": MockProvider,
        "rss": MockProvider,
        "google": MockProvider,
        "bing": MockProvider,
        "newscatcher": MockProvider,
    }
    monkeypatch.setattr(svc, "PROVIDERS", mock_providers)
    return MockProvider()


@pytest.fixture()
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# Test /scrape endpoint
def test_scrape_success(client, mock_provider, temp_dir):
    """Test successful scraping with valid provider"""
    save_path = temp_dir / "results.jsonl"
    payload = {
        "keywords": "artificial intelligence",
        "provider": "google",
        "after": "2024-10-01",
        "before": "2024-10-20",
        "max_results": 10,
        "save_path": str(save_path),
        "language": "en",
    }
    
    response = client.post("/scrape", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["article_count"] == 2
    assert data["stored_path"] == str(save_path)


def test_scrape_invalid_provider(client, temp_dir):
    """Test scraping with invalid provider"""
    payload = {
        "keywords": "test",
        "provider": "invalid_provider",
        "max_results": 10,
    }
    
    response = client.post("/scrape", json=payload)
    assert response.status_code == 400
    assert "Unknown provider" in response.json()["detail"]


def test_scrape_minimal_params(client, mock_provider):
    """Test scraping with minimal required parameters"""
    payload = {
        "keywords": "test query",
    }
    
    response = client.post("/scrape", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["article_count"] == 2


def test_scrape_different_providers(client, mock_provider, temp_dir):
    """Test scraping with different providers"""
    providers = ["arxiv", "atom", "rss", "google", "bing", "newscatcher"]
    
    for provider in providers:
        save_path = temp_dir / f"results_{provider}.jsonl"
        payload = {
            "keywords": "test",
            "provider": provider,
            "max_results": 5,
            "save_path": str(save_path),
        }
        
        response = client.post("/scrape", json=payload)
        assert response.status_code == 200, f"Failed for provider: {provider}"
        data = response.json()
        assert data["article_count"] == 2


# Test /auto-scrape endpoint
def test_auto_scrape_success(client, mock_provider, temp_dir):
    """Test successful auto-scraping with requests file"""
    # Create a requests file
    requests_file = temp_dir / "requests.txt"
    with open(requests_file, "w") as f:
        f.write("query1;2024-10-01;2024-10-10\n")
        f.write("query2;2024-10-11;2024-10-20\n")
    
    save_path = temp_dir / "auto_results.jsonl"
    payload = {
        "requests_file": str(requests_file),
        "max_results": 10,
        "provider": "google",
        "save_path": str(save_path),
        "language": "en",
        "evaluate_articles_quality": False,
        "minimum_quality_level": "AVERAGE",
    }
    
    response = client.post("/auto-scrape", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["article_count"] == 4  # 2 articles * 2 queries


def test_auto_scrape_invalid_file(client, mock_provider, temp_dir):
    """Test auto-scraping with non-existent file"""
    payload = {
        "requests_file": str(temp_dir / "nonexistent.txt"),
        "max_results": 10,
        "provider": "google",
    }
    
    response = client.post("/auto-scrape", json=payload)
    assert response.status_code == 400
    assert "Bad file format" in response.json()["detail"]


def test_auto_scrape_invalid_provider(client, temp_dir):
    """Test auto-scraping with invalid provider"""
    requests_file = temp_dir / "requests.txt"
    with open(requests_file, "w") as f:
        f.write("query1;2024-10-01;2024-10-10\n")
    
    payload = {
        "requests_file": str(requests_file),
        "provider": "invalid_provider",
    }
    
    response = client.post("/auto-scrape", json=payload)
    assert response.status_code == 400
    assert "Unknown provider" in response.json()["detail"]


# Test /generate-query-file endpoint
def test_generate_query_file_success(client, temp_dir):
    """Test successful query file generation"""
    save_path = temp_dir / "queries.txt"
    payload = {
        "keywords": "machine learning",
        "after": "2024-10-01",
        "before": "2024-10-31",
        "save_path": str(save_path),
        "interval": 10,
    }
    
    response = client.post("/generate-query-file", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["save_path"] == str(save_path)
    assert data["line_count"] == 3  # 30 days / 10 day interval
    
    # Verify file content
    assert save_path.exists()
    with open(save_path) as f:
        lines = f.readlines()
        assert len(lines) == 3
        assert "machine learning" in lines[0]


def test_generate_query_file_default_interval(client, temp_dir):
    """Test query file generation with default interval"""
    save_path = temp_dir / "queries.txt"
    payload = {
        "keywords": "test",
        "after": "2024-10-01",
        "before": "2024-10-31",
        "save_path": str(save_path),
    }
    
    response = client.post("/generate-query-file", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["line_count"] == 1  # 30 days / 30 day default interval


def test_generate_query_file_small_interval(client, temp_dir):
    """Test query file generation with small interval"""
    save_path = temp_dir / "queries.txt"
    payload = {
        "keywords": "test",
        "after": "2024-10-01",
        "before": "2024-10-11",
        "save_path": str(save_path),
        "interval": 1,
    }
    
    response = client.post("/generate-query-file", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["line_count"] == 10  # 10 days / 1 day interval


# Test /scrape-feed endpoint
def test_scrape_feed_success_google_provider(client, mock_provider, temp_dir, monkeypatch):
    """Test successful feed scraping with Google provider"""
    # Create a mock feed config file
    feed_cfg = temp_dir / "feed.toml"
    feed_cfg_content = """
[data-feed]
id = "test_feed"
query = "climate change"
provider = "google"
max_results = 10
number_of_days = 7
language = "en"
feed_dir_path = "feeds/test"
evaluate_articles_quality = false
minimum_quality_level = "AVERAGE"
"""
    with open(feed_cfg, "w") as f:
        f.write(feed_cfg_content)
    
    # Mock load_toml_config
    def mock_load_toml(path):
        return {
            "data-feed": {
                "id": "test_feed",
                "query": "climate change",
                "provider": "google",
                "max_results": 10,
                "number_of_days": 7,
                "language": "en",
                "feed_dir_path": "feeds/test",
                "evaluate_articles_quality": False,
                "minimum_quality_level": "AVERAGE",
            }
        }
    
    monkeypatch.setattr(svc, "load_toml_config", mock_load_toml)
    monkeypatch.setattr(svc, "FEED_BASE_PATH", temp_dir)
    
    payload = {"feed_cfg": str(feed_cfg)}
    response = client.post("/scrape-feed", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["article_count"] > 0


def test_scrape_feed_success_arxiv_provider(client, mock_provider, temp_dir, monkeypatch):
    """Test successful feed scraping with arxiv provider (batch provider)"""
    feed_cfg = temp_dir / "feed.toml"
    
    def mock_load_toml(path):
        return {
            "data-feed": {
                "id": "test_feed",
                "query": "quantum computing",
                "provider": "arxiv",
                "max_results": 10,
                "number_of_days": 7,
                "language": "en",
                "feed_dir_path": "feeds/test",
            }
        }
    
    monkeypatch.setattr(svc, "load_toml_config", mock_load_toml)
    monkeypatch.setattr(svc, "FEED_BASE_PATH", temp_dir)
    
    payload = {"feed_cfg": str(feed_cfg)}
    response = client.post("/scrape-feed", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["article_count"] > 0


# Test /schedule-scrapping endpoint
def test_schedule_scrapping_success(client, temp_dir, monkeypatch):
    """Test successful scheduling of scrapping"""
    feed_cfg = temp_dir / "feed.toml"
    feed_cfg.write_text("[data-feed]\nid = 'test'")
    
    # Mock scheduler_utils
    mock_scheduler = MagicMock()
    monkeypatch.setattr(svc, "scheduler_utils", mock_scheduler)
    
    payload = {"feed_cfg": str(feed_cfg)}
    response = client.post("/schedule-scrapping", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "scheduled"
    mock_scheduler.schedule_scrapping.assert_called_once()


def test_schedule_scrapping_error(client, temp_dir, monkeypatch):
    """Test scheduling error handling"""
    feed_cfg = temp_dir / "feed.toml"
    
    # Mock scheduler_utils to raise an exception
    mock_scheduler = MagicMock()
    mock_scheduler.schedule_scrapping.side_effect = Exception("Scheduling failed")
    monkeypatch.setattr(svc, "scheduler_utils", mock_scheduler)
    
    payload = {"feed_cfg": str(feed_cfg)}
    response = client.post("/schedule-scrapping", json=payload)
    assert response.status_code == 500
    assert "Scheduling failed" in response.json()["detail"]


# Edge case tests
def test_scrape_with_empty_keywords(client, mock_provider):
    """Test scraping with empty keywords"""
    payload = {
        "keywords": "",
        "provider": "google",
    }
    
    response = client.post("/scrape", json=payload)
    # Should still work, provider will handle empty query
    assert response.status_code == 200


def test_scrape_with_date_range(client, mock_provider, temp_dir):
    """Test scraping with specific date range"""
    save_path = temp_dir / "dated_results.jsonl"
    payload = {
        "keywords": "test",
        "provider": "google",
        "after": "2024-01-01",
        "before": "2024-12-31",
        "max_results": 50,
        "save_path": str(save_path),
    }
    
    response = client.post("/scrape", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["article_count"] == 2


def test_scrape_max_results_limit(client, mock_provider):
    """Test scraping respects max_results parameter"""
    payload = {
        "keywords": "test",
        "provider": "google",
        "max_results": 1,
    }
    
    response = client.post("/scrape", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["article_count"] == 1


def test_generate_query_file_same_dates(client, temp_dir):
    """Test query file generation with same start and end dates"""
    save_path = temp_dir / "queries.txt"
    payload = {
        "keywords": "test",
        "after": "2024-10-01",
        "before": "2024-10-01",
        "save_path": str(save_path),
        "interval": 1,
    }
    
    response = client.post("/generate-query-file", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["line_count"] == 0  # No intervals for same date


def test_auto_scrape_with_quality_evaluation(client, mock_provider, temp_dir):
    """Test auto-scraping with article quality evaluation enabled"""
    requests_file = temp_dir / "requests.txt"
    with open(requests_file, "w") as f:
        f.write("query1;2024-10-01;2024-10-10\n")
    
    save_path = temp_dir / "quality_results.jsonl"
    payload = {
        "requests_file": str(requests_file),
        "max_results": 10,
        "provider": "google",
        "save_path": str(save_path),
        "evaluate_articles_quality": True,
        "minimum_quality_level": "Good",
    }
    
    response = client.post("/auto-scrape", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["article_count"] >= 0  # May be filtered by quality
