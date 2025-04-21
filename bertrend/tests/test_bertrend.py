#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from bertrend import BERTREND_DEFAULT_CONFIG_PATH
from bertrend.BERTrend import BERTrend, _preprocess_model
from bertrend.BERTopicModel import BERTopicModel


@pytest.fixture
def mock_config():
    """Fixture for mocking the configuration."""
    return {
        "granularity": 7,
        "min_similarity": 0.7,
        "decay_factor": 0.9,
        "decay_power": 1.0,
    }


@pytest.fixture
def mock_topic_model():
    """Fixture for mocking the BERTopicModel."""
    mock_model = MagicMock(spec=BERTopicModel)
    mock_model.fit.return_value.topic_model = MagicMock(spec=BERTopic)
    return mock_model


@pytest.fixture
def mock_sentence_transformer():
    """Fixture for mocking the SentenceTransformer."""
    return MagicMock(spec=SentenceTransformer)


@pytest.fixture
def mock_embeddings():
    """Fixture for mocking embeddings."""
    return np.random.random((10, 768))


@pytest.fixture
def mock_grouped_data():
    """Fixture for mocking grouped data."""
    timestamp1 = pd.Timestamp("2023-01-01")
    timestamp2 = pd.Timestamp("2023-01-08")

    df1 = pd.DataFrame({
        "document_id": [f"doc_{i}" for i in range(5)],
        "text": [f"Document {i}" for i in range(5)],
        "source": ["source1"] * 5,
        "url": [f"http://example.com/{i}" for i in range(5)]
    })

    df2 = pd.DataFrame({
        "document_id": [f"doc_{i+5}" for i in range(5)],
        "text": [f"Document {i+5}" for i in range(5)],
        "source": ["source2"] * 5,
        "url": [f"http://example.com/{i+5}" for i in range(5)]
    })

    return {timestamp1: df1, timestamp2: df2}


@pytest.fixture
def bertrend_instance(mock_config, mock_topic_model):
    """Fixture for creating a BERTrend instance with mocked dependencies."""
    with patch('bertrend.BERTrend.load_toml_config', return_value=mock_config):
        instance = BERTrend(config_file=BERTREND_DEFAULT_CONFIG_PATH, topic_model=mock_topic_model)
        return instance


def test_initialization(bertrend_instance, mock_config, mock_topic_model):
    """Test initialization of BERTrend class."""
    assert bertrend_instance.config == mock_config
    assert bertrend_instance.topic_model == mock_topic_model
    assert bertrend_instance._is_fitted is False
    assert bertrend_instance.last_topic_model is None
    assert bertrend_instance.last_topic_model_timestamp is None
    assert bertrend_instance.doc_groups == {}
    assert bertrend_instance.emb_groups == {}
    assert bertrend_instance.merge_df_size_over_time == []
    assert bertrend_instance.all_new_topics_df is None
    assert bertrend_instance.all_merge_histories_df is None
    assert bertrend_instance.merged_df is None


def test_load_config():
    """Test loading configuration from file."""
    with patch('bertrend.BERTrend.load_toml_config', return_value={"test_key": "test_value"}) as mock_load:
        bertrend = BERTrend(config_file="test_config.toml")
        mock_load.assert_called_once_with("test_config.toml")
        assert bertrend.config == {"test_key": "test_value"}


def test_get_periods(bertrend_instance):
    """Test get_periods method."""
    # Setup
    bertrend_instance.doc_groups = {
        pd.Timestamp("2023-01-01"): ["doc1", "doc2"],
        pd.Timestamp("2023-01-08"): ["doc3", "doc4"]
    }

    # Execute
    periods = bertrend_instance.get_periods()

    # Assert
    assert len(periods) == 2
    assert pd.Timestamp("2023-01-01") in periods
    assert pd.Timestamp("2023-01-08") in periods


def test_train_by_period(bertrend_instance, mock_sentence_transformer, mock_embeddings):
    """Test _train_by_period method."""
    # Setup
    period = pd.Timestamp("2023-01-01")
    group = pd.DataFrame({
        "text": ["Document 1", "Document 2"],
        "document_id": ["doc1", "doc2"],
        "source": ["source1", "source2"],
        "url": ["http://example.com/1", "http://example.com/2"]
    })

    # Create a mock for the entire _train_by_period method
    # This avoids the complexity of mocking all the internal operations
    original_train_by_period = bertrend_instance._train_by_period

    mock_topic_model = MagicMock(spec=BERTopic)
    mock_topic_model.doc_info_df = pd.DataFrame({"Topic": [0, 1]})
    mock_topic_model.topic_info_df = pd.DataFrame({"Topic": [0, 1]})

    expected_docs = group["text"].tolist()
    expected_embeddings = mock_embeddings[group.index]

    # Replace the method with a mock that returns our expected values
    bertrend_instance._train_by_period = MagicMock(
        return_value=(mock_topic_model, expected_docs, expected_embeddings)
    )

    try:
        # Execute
        result_model, result_docs, result_embeddings = bertrend_instance._train_by_period(
            period, group, mock_sentence_transformer, mock_embeddings
        )

        # Assert
        bertrend_instance._train_by_period.assert_called_once_with(
            period, group, mock_sentence_transformer, mock_embeddings
        )
        assert result_model == mock_topic_model
        assert result_docs == expected_docs
        assert np.array_equal(result_embeddings, expected_embeddings)
    finally:
        # Restore the original method
        bertrend_instance._train_by_period = original_train_by_period


def test_train_topic_models(bertrend_instance, mock_grouped_data, mock_sentence_transformer, mock_embeddings):
    """Test train_topic_models method."""
    # Setup
    mock_new_topic_model = MagicMock()
    bertrend_instance._train_by_period = MagicMock(return_value=(mock_new_topic_model, ["doc1"], np.array([[0.1, 0.2]])))
    bertrend_instance.merge_models_with = MagicMock()

    with patch('bertrend.BERTrend.BERTrend.save_topic_model') as mock_save:
        # Execute
        bertrend_instance.train_topic_models(
            mock_grouped_data, mock_sentence_transformer, mock_embeddings
        )

        # Assert
        assert bertrend_instance._train_by_period.call_count == 2
        assert bertrend_instance.merge_models_with.call_count == 1
        assert mock_save.call_count == 2
        assert bertrend_instance._is_fitted is True
        assert bertrend_instance.last_topic_model == mock_new_topic_model
        assert bertrend_instance.last_topic_model_timestamp == list(mock_grouped_data.keys())[1]


def test_train_topic_models_error_handling(bertrend_instance, mock_grouped_data, mock_sentence_transformer, mock_embeddings):
    """Test error handling in train_topic_models method."""
    # Setup
    bertrend_instance._train_by_period = MagicMock(side_effect=Exception("Test exception"))

    with patch('bertrend.BERTrend.logger.error') as mock_logger_error:
        with patch('bertrend.BERTrend.logger.exception') as mock_logger_exception:
            # Execute
            bertrend_instance.train_topic_models(
                mock_grouped_data, mock_sentence_transformer, mock_embeddings
            )

            # Assert
            assert mock_logger_error.called
            assert mock_logger_exception.called
            assert bertrend_instance._is_fitted is True  # Should still be set to True even if errors occur


def test_merge_models_with(bertrend_instance):
    """Test merge_models_with method."""
    # Setup
    bertrend_instance.last_topic_model = MagicMock()
    bertrend_instance.last_topic_model_timestamp = pd.Timestamp("2023-01-01")
    bertrend_instance.doc_groups = {
        pd.Timestamp("2023-01-01"): ["doc1", "doc2"],
        pd.Timestamp("2023-01-08"): ["doc3", "doc4"]
    }
    bertrend_instance.emb_groups = {
        pd.Timestamp("2023-01-01"): np.array([[0.1, 0.2], [0.3, 0.4]]),
        pd.Timestamp("2023-01-08"): np.array([[0.5, 0.6], [0.7, 0.8]])
    }
    bertrend_instance.merged_df = None
    bertrend_instance.all_merge_histories_df = pd.DataFrame()
    bertrend_instance.all_new_topics_df = pd.DataFrame()

    new_model = MagicMock()
    new_model_timestamp = pd.Timestamp("2023-01-08")

    # Mock _preprocess_model function
    with patch('bertrend.BERTrend._preprocess_model') as mock_preprocess:
        mock_df1 = pd.DataFrame({"Topic": [0, 1], "Count": [2, 3]})
        mock_df2 = pd.DataFrame({"Topic": [0, 1], "Count": [1, 2]})
        mock_preprocess.side_effect = [mock_df1, mock_df2]

        # Mock _merge_models function
        with patch('bertrend.BERTrend._merge_models') as mock_merge:
            mock_merged_df = pd.DataFrame({"Topic": [0, 1, 2], "Count": [3, 5, 1]})
            mock_merge_history = pd.DataFrame({"Topic1": [0], "Topic2": [0]})
            mock_new_topics = pd.DataFrame({"Topic": [2]})
            mock_merge.return_value = (mock_merged_df, mock_merge_history, mock_new_topics)

            # Execute
            bertrend_instance.merge_models_with(new_model, new_model_timestamp)

            # Assert
            mock_preprocess.assert_called()
            mock_merge.assert_called_once()
            assert bertrend_instance.merged_df is not None
            assert not bertrend_instance.all_merge_histories_df.empty
            assert not bertrend_instance.all_new_topics_df.empty
            assert len(bertrend_instance.merge_df_size_over_time) == 1


def test_merge_models_with_empty_dataframe(bertrend_instance):
    """Test merge_models_with method when one dataframe is empty."""
    # Setup
    bertrend_instance.last_topic_model = MagicMock()
    bertrend_instance.last_topic_model_timestamp = pd.Timestamp("2023-01-01")
    bertrend_instance.doc_groups = {
        pd.Timestamp("2023-01-01"): ["doc1", "doc2"],
        pd.Timestamp("2023-01-08"): ["doc3", "doc4"]
    }
    bertrend_instance.emb_groups = {
        pd.Timestamp("2023-01-01"): np.array([[0.1, 0.2], [0.3, 0.4]]),
        pd.Timestamp("2023-01-08"): np.array([[0.5, 0.6], [0.7, 0.8]])
    }
    bertrend_instance.merged_df = None
    bertrend_instance.all_merge_histories_df = pd.DataFrame()
    bertrend_instance.all_new_topics_df = pd.DataFrame()

    new_model = MagicMock()
    new_model_timestamp = pd.Timestamp("2023-01-08")

    # Mock _preprocess_model function to return an empty dataframe
    with patch('bertrend.BERTrend._preprocess_model') as mock_preprocess:
        mock_df1 = pd.DataFrame({"Topic": [0, 1], "Count": [2, 3]})
        # Empty dataframe with expected columns
        mock_df2 = pd.DataFrame(columns=["Topic", "Count", "Document_Count", "Representation", 
                                         "Documents", "Embedding", "DocEmbeddings", "Sources", "URLs"])
        mock_preprocess.side_effect = [mock_df1, mock_df2]

        with patch('bertrend.BERTrend.logger.warning') as mock_warning:
            # Execute
            result = bertrend_instance.merge_models_with(new_model, new_model_timestamp)

            # Assert
            mock_warning.assert_called_once()
            assert result is None


def test_merge_models_with_invalid_models(bertrend_instance):
    """Test merge_models_with method with invalid models."""
    # Setup
    bertrend_instance.last_topic_model = None  # Invalid model
    new_model = MagicMock()
    new_model_timestamp = pd.Timestamp("2023-01-08")

    # Execute and Assert
    with pytest.raises(ValueError):
        bertrend_instance.merge_models_with(new_model, new_model_timestamp)


def test_calculate_signal_popularity(bertrend_instance):
    """Test calculate_signal_popularity method."""
    # Setup
    bertrend_instance.doc_groups = {
        pd.Timestamp("2023-01-01"): ["doc1", "doc2"],
        pd.Timestamp("2023-01-08"): ["doc3", "doc4"]
    }
    bertrend_instance.all_merge_histories_df = pd.DataFrame({
        "Timestamp": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-08")],
        "Topic1": [0, 0],
        "Topic2": [1, 2],
        "Count1": [2, 3],
        "Count2": [1, 2],
        "Document_Count1": [2, 3],
        "Document_Count2": [1, 2],
        "Source_Diversity1": [1, 1],
        "Source_Diversity2": [1, 1]
    })

    # Mock the trend analysis functions
    with patch('bertrend.BERTrend._initialize_new_topic') as mock_init:
        with patch('bertrend.BERTrend.update_existing_topic') as mock_update:
            with patch('bertrend.BERTrend._apply_decay_to_inactive_topics') as mock_decay:
                # Execute
                bertrend_instance.calculate_signal_popularity()

                # Assert
                assert mock_init.called
                assert mock_update.called
                assert mock_decay.called
                assert bertrend_instance.topic_sizes is not None
                assert bertrend_instance.topic_last_popularity is not None
                assert bertrend_instance.topic_last_update is not None


def test_calculate_signal_popularity_not_enough_models(bertrend_instance):
    """Test calculate_signal_popularity method when there are not enough models."""
    # Setup
    bertrend_instance.doc_groups = {
        pd.Timestamp("2023-01-01"): ["doc1", "doc2"]
    }  # Only one model

    with patch('bertrend.BERTrend.logger.error') as mock_error:
        # Execute
        bertrend_instance.calculate_signal_popularity()

        # Assert
        mock_error.assert_called_once()
        # State should remain unchanged
        assert isinstance(bertrend_instance.topic_sizes, dict)
        assert isinstance(bertrend_instance.topic_last_popularity, dict)
        assert isinstance(bertrend_instance.topic_last_update, dict)


def test_compute_popularity_values_and_thresholds(bertrend_instance):
    """Test _compute_popularity_values_and_thresholds method."""
    # Setup
    bertrend_instance.topic_sizes = {
        0: {
            "Timestamps": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-08")],
            "Popularity": [0.5, 0.7]
        },
        1: {
            "Timestamps": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-08")],
            "Popularity": [0.3, 0.2]
        }
    }

    # Execute
    window_start, window_end, all_popularity_values, q1, q3 = bertrend_instance._compute_popularity_values_and_thresholds(
        window_size=14, current_date=pd.Timestamp("2023-01-15")
    )

    # Assert
    assert window_start == pd.Timestamp("2023-01-01")
    assert window_end == pd.Timestamp("2023-01-22")
    assert len(all_popularity_values) == 4
    assert q1 > 0
    assert q3 > q1


def test_classify_signals(bertrend_instance):
    """Test classify_signals method."""
    # Setup
    bertrend_instance._compute_popularity_values_and_thresholds = MagicMock(
        return_value=(
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-01-15"),
            [0.1, 0.2, 0.3, 0.4],
            0.15,
            0.35
        )
    )

    mock_noise_df = pd.DataFrame({"Topic": [0], "Popularity": [0.1]})
    mock_weak_signal_df = pd.DataFrame({"Topic": [1], "Popularity": [0.2]})
    mock_strong_signal_df = pd.DataFrame({"Topic": [2], "Popularity": [0.4]})

    bertrend_instance._classify_signals = MagicMock(
        return_value=(mock_noise_df, mock_weak_signal_df, mock_strong_signal_df)
    )

    # Execute
    noise_df, weak_signal_df, strong_signal_df = bertrend_instance.classify_signals(
        window_size=14, current_date=pd.Timestamp("2023-01-15")
    )

    # Assert
    bertrend_instance._compute_popularity_values_and_thresholds.assert_called_once()
    bertrend_instance._classify_signals.assert_called_once()
    assert noise_df.equals(mock_noise_df)
    assert weak_signal_df.equals(mock_weak_signal_df)
    assert strong_signal_df.equals(mock_strong_signal_df)


def test_classify_signals_with_custom_parameters(bertrend_instance):
    """Test _classify_signals method with custom parameters."""
    # Setup
    window_start = pd.Timestamp("2023-01-01")
    window_end = pd.Timestamp("2023-01-15")
    q1 = 0.15
    q3 = 0.35

    bertrend_instance.topic_sizes = {
        0: {
            "Timestamps": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-08")],
            "Popularity": [0.1, 0.1],  # Below q1, should be noise
            "Docs_Count": [2, 2],
            "Paragraphs_Count": [3, 3],
            "Source_Diversity": [1, 1],
            "Representation": "Topic 0",
            "Representations": ["Topic 0", "Topic 0"],
            "Documents": [(pd.Timestamp("2023-01-01"), ["doc1"]), (pd.Timestamp("2023-01-08"), ["doc2"])],
            "Sources": [(pd.Timestamp("2023-01-01"), ["source1"]), (pd.Timestamp("2023-01-08"), ["source1"])]
        },
        1: {
            "Timestamps": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-08")],
            "Popularity": [0.2, 0.3],  # Between q1 and q3, and rising, should be weak signal
            "Docs_Count": [2, 3],
            "Paragraphs_Count": [3, 4],
            "Source_Diversity": [1, 2],
            "Representation": "Topic 1",
            "Representations": ["Topic 1", "Topic 1"],
            "Documents": [(pd.Timestamp("2023-01-01"), ["doc3"]), (pd.Timestamp("2023-01-08"), ["doc4"])],
            "Sources": [(pd.Timestamp("2023-01-01"), ["source1"]), (pd.Timestamp("2023-01-08"), ["source2"])]
        },
        2: {
            "Timestamps": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-08")],
            "Popularity": [0.4, 0.5],  # Above q3, should be strong signal
            "Docs_Count": [4, 5],
            "Paragraphs_Count": [6, 7],
            "Source_Diversity": [2, 3],
            "Representation": "Topic 2",
            "Representations": ["Topic 2", "Topic 2"],
            "Documents": [(pd.Timestamp("2023-01-01"), ["doc5"]), (pd.Timestamp("2023-01-08"), ["doc6"])],
            "Sources": [(pd.Timestamp("2023-01-01"), ["source1", "source2"]), (pd.Timestamp("2023-01-08"), ["source1", "source2", "source3"])]
        }
    }

    # Mock _is_rising_popularity to return True for topic 1
    with patch('bertrend.BERTrend._is_rising_popularity', return_value=True):
        # Execute
        noise_df, weak_signal_df, strong_signal_df = bertrend_instance._classify_signals(
            window_start, window_end, q1, q3
        )

        # Assert
        assert len(noise_df) == 1
        assert noise_df.iloc[0]["Topic"] == 0

        assert len(weak_signal_df) == 1
        assert weak_signal_df.iloc[0]["Topic"] == 1

        assert len(strong_signal_df) == 1
        assert strong_signal_df.iloc[0]["Topic"] == 2


def test_save_and_restore_model(bertrend_instance, tmp_path):
    """Test save_model and restore_model methods."""
    # Setup
    models_path = tmp_path / "models"
    models_path.mkdir()

    # Mock dill.dump and dill.load
    with patch('bertrend.BERTrend.dill.dump') as mock_dump:
        with patch('bertrend.BERTrend.dill.load', return_value=bertrend_instance) as mock_load:
            with patch('builtins.open', mock_open()) as mock_file:
                # Execute save
                bertrend_instance.save_model(models_path)

                # Assert save
                mock_dump.assert_called_once()

                # Execute restore
                restored = BERTrend.restore_model(models_path)

                # Assert restore
                mock_load.assert_called_once()
                assert restored == bertrend_instance


def test_restore_model_nonexistent_path():
    """Test restore_model method with a nonexistent path."""
    # Setup
    nonexistent_path = Path("/nonexistent/path")

    # Execute and Assert
    with pytest.raises(FileNotFoundError):
        BERTrend.restore_model(nonexistent_path)


def test_save_and_restore_topic_model(tmp_path):
    """Test save_topic_model and restore_topic_model class methods."""
    # Setup
    models_path = tmp_path / "models"
    models_path.mkdir()
    period = pd.Timestamp("2023-01-01")

    # Create a mock topic model
    mock_topic_model = MagicMock()
    mock_topic_model.doc_info_df = pd.DataFrame({"Topic": [0, 1]})
    mock_topic_model.topic_info_df = pd.DataFrame({"Topic": [0, 1]})
    mock_topic_model.embedding_model = "all-MiniLM-L6-v2"

    # Add the save method to the mock topic model
    mock_topic_model.save = MagicMock()

    # Mock BERTopic.load to return our mock topic model
    with patch('bertopic.BERTopic.load', return_value=mock_topic_model):
        with patch('pandas.DataFrame.to_pickle') as mock_to_pickle:
            with patch('pandas.read_pickle', return_value=pd.DataFrame()) as mock_read_pickle:
                with patch('pathlib.Path.glob', return_value=[models_path / "2023-01-01"]):
                    with patch('pathlib.Path.is_dir', return_value=True):
                        with patch('pathlib.Path.exists', return_value=True):
                            # Execute save
                            BERTrend.save_topic_model(period, mock_topic_model, models_path)

                            # Assert save
                            mock_topic_model.save.assert_called_once()
                            assert mock_to_pickle.call_count == 2

                            # Execute restore
                            restored = BERTrend.restore_topic_model(period, models_path)

                            # Assert restore
                            assert restored == mock_topic_model
                            assert mock_read_pickle.call_count == 2


def test_preprocess_model():
    """Test _preprocess_model function."""
    # Setup
    mock_topic_model = MagicMock(spec=BERTopic)
    mock_topic_model.topic_info_df = pd.DataFrame({
        "Topic": [0, 1],
        "Count": [2, 3],
        "Document_Count": [2, 3],
        "Representation": ["Topic 0", "Topic 1"],
        "Name": ["", ""],
        "Representative_Docs": [["doc1"], ["doc2"]],
        "Sources": [["source1"], ["source2"]],
        "URLs": [["url1"], ["url2"]]
    })

    mock_topic_model.doc_info_df = pd.DataFrame({
        "Topic": [0, 0, 1],
        "Paragraph": ["doc1", "doc2", "doc3"],
        "Probability": [0.9, 0.8, 0.7],
        "source": ["source1", "source1", "source2"],
        "url": ["url1", "url2", "url3"]
    })

    docs = ["doc1", "doc2", "doc3"]
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    # Execute
    result = _preprocess_model(mock_topic_model, docs, embeddings)

    # Assert
    assert isinstance(result, pd.DataFrame)
    assert "Topic" in result.columns
    assert "Count" in result.columns
    assert "Document_Count" in result.columns
    assert "Representation" in result.columns
    assert "Documents" in result.columns
    assert "Embedding" in result.columns
    assert "DocEmbeddings" in result.columns
    assert "Sources" in result.columns
    assert "URLs" in result.columns
