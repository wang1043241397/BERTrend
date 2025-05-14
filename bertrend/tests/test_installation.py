"""
Test script to verify that the installation process works correctly with the new dependency management structure.
This script attempts to import key components from each dependency group.
"""


def test_core_dependencies():
    """Test that core dependencies for topic modeling and analysis are installed correctly."""
    try:
        import bertopic
        import gensim
        import hdbscan
        import numpy
        import pandas
        import sklearn
        import scipy
        import sentence_transformers
        import torch
        import umap

        print("✅ Core dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import core dependencies: {e}")
        return False


def test_nlp_dependencies():
    """Test that NLP and text processing dependencies are installed correctly."""
    try:
        import langdetect
        import lxml_html_clean
        import markdown
        import nltk
        import sentencepiece
        import thefuzz
        import tldextract
        import tokenizers

        print("✅ NLP dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import NLP dependencies: {e}")
        return False


def test_llm_dependencies():
    """Test that LLM integration dependencies are installed correctly."""
    try:
        import langchain_core
        import openai
        import tiktoken

        print("✅ LLM dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import LLM dependencies: {e}")
        return False


def test_visualization_dependencies():
    """Test that visualization and UI dependencies are installed correctly."""
    try:
        import datamapplot
        import plotly
        import plotly_resampler
        import pylabeladjust
        import pyqtree
        import seaborn
        import streamlit

        print("✅ Visualization dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import visualization dependencies: {e}")
        return False


def test_utility_dependencies():
    """Test that utility dependencies are installed correctly."""
    try:
        import black
        import cron_descriptor
        import dask
        import dateparser
        import dill
        import joblib
        import jsonlines
        import loguru
        import opentelemetry
        import requests
        import tqdm

        print("✅ Utility dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import utility dependencies: {e}")
        return False


def test_optional_test_dependencies():
    """Test that optional test dependencies are installed correctly."""
    try:
        import pytest
        import coverage

        print("✅ Test dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import test dependencies: {e}")
        return False


def test_optional_apps_dependencies():
    """Test that optional apps dependencies are installed correctly."""
    try:
        # Only test a subset of apps dependencies to avoid errors if some are not installed
        import arxiv
        import feedparser
        import googleapiclient

        print("✅ Apps dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import apps dependencies: {e}")
        return False


if __name__ == "__main__":
    print("Testing BERTrend installation...")

    # Test core dependencies
    core_ok = test_core_dependencies()

    # Test NLP dependencies
    nlp_ok = test_nlp_dependencies()

    # Test LLM dependencies
    llm_ok = test_llm_dependencies()

    # Test visualization dependencies
    viz_ok = test_visualization_dependencies()

    # Test utility dependencies
    util_ok = test_utility_dependencies()

    # Test optional dependencies
    test_ok = test_optional_test_dependencies()
    apps_ok = test_optional_apps_dependencies()

    # Print summary
    print("\nInstallation Test Summary:")
    print(f"Core dependencies: {'✅' if core_ok else '❌'}")
    print(f"NLP dependencies: {'✅' if nlp_ok else '❌'}")
    print(f"LLM dependencies: {'✅' if llm_ok else '❌'}")
    print(f"Visualization dependencies: {'✅' if viz_ok else '❌'}")
    print(f"Utility dependencies: {'✅' if util_ok else '❌'}")
    print(f"Test dependencies: {'✅' if test_ok else '❌'}")
    print(f"Apps dependencies: {'✅' if apps_ok else '❌'}")

    if all([core_ok, nlp_ok, llm_ok, viz_ok, util_ok]):
        print("\n✅ Core installation is complete and working correctly.")
    else:
        print("\n❌ Some core dependencies are missing. Please check the output above.")

    if test_ok and apps_ok:
        print("✅ All optional dependencies are installed correctly.")
    elif test_ok:
        print("✅ Test dependencies are installed correctly.")
        print(
            "❌ Apps dependencies are missing. Install with 'pip install \".[apps]\"' or 'poetry install --with apps'"
        )
    elif apps_ok:
        print("✅ Apps dependencies are installed correctly.")
        print(
            "❌ Test dependencies are missing. Install with 'pip install \".[tests]\"' or 'poetry install --with test'"
        )
    else:
        print(
            "❌ Optional dependencies are missing. Install with 'pip install \".[tests,apps]\"' or 'poetry install --with test,apps'"
        )
