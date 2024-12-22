#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import streamlit as st
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


def display_bertopic_hyperparameters():
    with st.expander("Embedding Model Settings", expanded=False):
        language = st.selectbox("Select Language", LANGUAGES, key="language")
        embedding_dtype = st.selectbox(
            "Embedding Dtype", EMBEDDING_DTYPES, key="embedding_dtype"
        )

        embedding_models = (
            ENGLISH_EMBEDDING_MODELS
            if language == "English"
            else FRENCH_EMBEDDING_MODELS
        )
        embedding_model_name = st.selectbox(
            "Embedding Model", embedding_models, key="embedding_model_name"
        )

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
                st.number_input(
                    label,
                    value=default,
                    min_value=min_val,
                    max_value=max_val,
                    key=key,
                )

            if expander == "HDBSCAN Hyperparameters":
                st.selectbox(
                    "Cluster Selection Method",
                    HDBSCAN_CLUSTER_SELECTION_METHODS,
                    key="hdbscan_cluster_selection_method",
                )
            elif expander == "Vectorizer Hyperparameters":
                st.selectbox(
                    "N-Gram range",
                    VECTORIZER_NGRAM_RANGES,
                    key="vectorizer_ngram_range",
                )

    with st.expander("Merging Hyperparameters", expanded=False):
        st.slider(
            "Minimum Similarity for Merging",
            0.0,
            1.0,
            DEFAULT_MIN_SIMILARITY,
            0.01,
            key="min_similarity",
        )

    with st.expander("Zero-shot Parameters", expanded=False):
        st.slider(
            "Zeroshot Minimum Similarity",
            0.0,
            1.0,
            DEFAULT_ZEROSHOT_MIN_SIMILARITY,
            0.01,
            key="zeroshot_min_similarity",
        )
