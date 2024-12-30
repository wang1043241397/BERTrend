#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import streamlit as st

from bertrend import EMBEDDING_CONFIG
from bertrend.demos.demos_utils.session_state_manager import SessionStateManager
from bertrend.demos.demos_utils.state_utils import register_widget, save_widget_state
from bertrend.parameters import (
    DEFAULT_UMAP_N_COMPONENTS,
    DEFAULT_UMAP_N_NEIGHBORS,
    DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    DEFAULT_HDBSCAN_MIN_SAMPLES,
    DEFAULT_TOP_N_WORDS,
    DEFAULT_MIN_DF,
    DEFAULT_MIN_SIMILARITY,
    VECTORIZER_NGRAM_RANGES,
    DEFAULT_ZEROSHOT_MIN_SIMILARITY,
    HDBSCAN_CLUSTER_SELECTION_METHODS,
    EMBEDDING_DTYPES,
    LANGUAGES,
    ENGLISH_EMBEDDING_MODELS,
    FRENCH_EMBEDDING_MODELS,
)


def display_local_embeddings():
    register_widget("language")
    register_widget("embedding_dtype")
    register_widget("embedding_model_name")

    language = st.selectbox(
        "Select Language",
        LANGUAGES,
        key="language",
        on_change=save_widget_state,
    )
    st.selectbox(
        "Embedding Dtype",
        EMBEDDING_DTYPES,
        key="embedding_dtype",
        on_change=save_widget_state,
    )
    embedding_models = (
        ENGLISH_EMBEDDING_MODELS if language == "English" else FRENCH_EMBEDDING_MODELS
    )
    st.selectbox(
        "Embedding Model",
        embedding_models,
        key="embedding_model_name",
        on_change=save_widget_state,
    )


def display_remote_embeddings():
    register_widget("embedding_service_hostname")
    register_widget("embedding_service_port")
    if "embedding_service_hostname" not in st.session_state:
        st.session_state["embedding_service_hostname"] = EMBEDDING_CONFIG["host"]
    if "embedding_service_port" not in st.session_state:
        st.session_state["embedding_service_port"] = EMBEDDING_CONFIG["port"]
    st.text_input(
        "Embedding service hostname",
        key="embedding_service_hostname",
        on_change=save_widget_state,
    )
    st.text_input(
        "Embedding service port",
        value=EMBEDDING_CONFIG["port"],
        on_change=save_widget_state,
    )


def display_bertopic_hyperparameters():
    with st.expander("Embedding Model Settings", expanded=False):
        register_widget("embedding_service_type")
        if "embedding_service_type" not in st.session_state:
            st.session_state["embedding_service_type"] = "local"
        st.segmented_control(
            "Embedding service",
            selection_mode="single",
            key="embedding_service_type",
            options=["local", "remote"],
            on_change=save_widget_state,
        )
        if st.session_state["embedding_service_type"] == "local":
            display_local_embeddings()
        else:
            display_remote_embeddings()

    for expander, params in [
        (
            "UMAP Hyperparameters",
            [
                (
                    "umap_n_components",
                    "UMAP n_components",
                    DEFAULT_UMAP_N_COMPONENTS,
                    2,
                    100,
                ),
                (
                    "umap_n_neighbors",
                    "UMAP n_neighbors",
                    DEFAULT_UMAP_N_NEIGHBORS,
                    2,
                    100,
                ),
            ],
        ),
        (
            "HDBSCAN Hyperparameters",
            [
                (
                    "hdbscan_min_cluster_size",
                    "HDBSCAN min_cluster_size",
                    DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
                    2,
                    100,
                ),
                (
                    "hdbscan_min_samples",
                    "HDBSCAN min_sample",
                    DEFAULT_HDBSCAN_MIN_SAMPLES,
                    1,
                    100,
                ),
            ],
        ),
        (
            "Vectorizer Hyperparameters",
            [
                ("top_n_words", "Top N Words", DEFAULT_TOP_N_WORDS, 1, 50),
                ("min_df", "min_df", DEFAULT_MIN_DF, 1, 50),
            ],
        ),
    ]:
        with st.expander(expander, expanded=False):
            for key, label, default, min_val, max_val in params:
                register_widget(key)
                st.number_input(
                    label,
                    value=default,
                    min_value=min_val,
                    max_value=max_val,
                    key=key,
                    on_change=save_widget_state,
                )

            if expander == "HDBSCAN Hyperparameters":
                register_widget("hdbscan_cluster_selection_method")
                st.selectbox(
                    "Cluster Selection Method",
                    HDBSCAN_CLUSTER_SELECTION_METHODS,
                    key="hdbscan_cluster_selection_method",
                    on_change=save_widget_state,
                )
            elif expander == "Vectorizer Hyperparameters":
                register_widget("vectorizer_ngram_range")
                st.selectbox(
                    "N-Gram range",
                    VECTORIZER_NGRAM_RANGES,
                    key="vectorizer_ngram_range",
                    on_change=save_widget_state,
                )


def display_bertrend_hyperparameters():
    with st.expander("Merging Hyperparameters", expanded=False):
        register_widget("min_similarity")
        st.slider(
            "Minimum Similarity for Merging",
            0.0,
            1.0,
            DEFAULT_MIN_SIMILARITY,
            0.01,
            key="min_similarity",
            on_change=save_widget_state,
        )

    with st.expander("Zero-shot Parameters", expanded=False):
        register_widget("zeroshot_min_similarity")
        st.slider(
            "Zeroshot Minimum Similarity",
            0.0,
            1.0,
            DEFAULT_ZEROSHOT_MIN_SIMILARITY,
            0.01,
            key="zeroshot_min_similarity",
            on_change=save_widget_state,
        )
