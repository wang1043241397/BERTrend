#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
import os
from unittest.mock import patch, MagicMock
from openai import OpenAI, AzureOpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from bertrend.llm_utils.openai_client import OpenAI_Client


# Helper function to mock OpenAI responses
def mock_openai_response():
    mock_response = MagicMock(spec=ChatCompletion)
    mock_response.choices = [
        MagicMock(message=MagicMock(content="This is a mock response"))
    ]
    return mock_response


@pytest.fixture
def mock_api_key():
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    yield
    del os.environ["OPENAI_API_KEY"]


@pytest.fixture
def mock_azure_endpoint():
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    os.environ["OPENAI_ENDPOINT"] = "https://azure.com"
    yield
    del os.environ["OPENAI_API_KEY"]
    del os.environ["OPENAI_ENDPOINT"]


def test_initialization_with_azure(mock_azure_endpoint):
    """Test client initialization when using Azure endpoint"""
    client = OpenAI_Client(api_key="test_api_key", endpoint="https://azure.com")
    assert isinstance(client.llm_client, AzureOpenAI)


def test_initialization_without_azure(mock_api_key):
    """Test client initialization when using OpenAI endpoint"""
    client = OpenAI_Client(api_key="test_api_key")
    assert isinstance(client.llm_client, OpenAI)


def test_generate_user_prompt(mock_api_key):
    """Test generation with a simple user prompt"""
    client = OpenAI_Client(api_key="test_api_key")

    # Mocking the llm_client's chat completion method
    with patch.object(
        client.llm_client.chat.completions,
        "create",
        return_value=mock_openai_response(),
    ):
        result = client.generate("What is the weather today?")
        assert result == "This is a mock response"


def test_generate_from_history(mock_api_key):
    """Test generation using history of messages"""
    client = OpenAI_Client(api_key="test_api_key")

    # Mocking the llm_client's chat completion method
    with patch.object(
        client.llm_client.chat.completions,
        "create",
        return_value=mock_openai_response(),
    ):
        messages = [{"role": "user", "content": "What is the weather today?"}]
        result = client.generate_from_history(messages)
        assert result == "This is a mock response"


def test_api_error_handling(mock_api_key):
    """Test if an API error is properly handled"""
    client = OpenAI_Client(api_key="test_api_key")

    # Simulate an error during API call
    with patch.object(
        client.llm_client.chat.completions, "create", side_effect=Exception("API Error")
    ):
        result = client.generate("What is the weather today?")
        assert result == "OpenAI API fatal error: API Error"


def test_generate_with_streaming(mock_api_key):
    """Test if streaming works when 'stream' is True"""
    client = OpenAI_Client(api_key="test_api_key")

    # Mock streaming response
    mock_stream = MagicMock(spec=Stream)
    with patch.object(
        client.llm_client.chat.completions, "create", return_value=mock_stream
    ):
        result = client.generate("What is the weather today?", stream=True)
        assert result == mock_stream
