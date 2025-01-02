#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.


import numpy as np
import pandas as pd
import streamlit as st

from bertrend.utils.data_loading import (
    TIMESTAMP_COLUMN,
    GROUPED_TIMESTAMP_COLUMN,
    TEXT_COLUMN,
    TITLE_COLUMN,
    URL_COLUMN,
    CITATION_COUNT_COL,
)

# Default configuration parameters for the application
DEFAULT_PARAMETERS = {
    "embedding_model_name": "OrdalieTech/Solon-embeddings-base-0.1",
    "use_cached_embeddings": False,
    "bertopic_nr_topics": 0,
    "bertopic_top_n_words": 10,
    "umap_n_neighbors": 15,
    "umap_n_components": 5,
    "umap_min_dist": 0.0,
    "umap_metric": "cosine",
    "hdbscan_min_cluster_size": 10,
    "hdbscan_min_samples": 10,
    "hdbscan_metric": "euclidean",
    "hdbscan_cluster_selection_method": "eom",
    "hdbscan_cluster_selection_epsilon": 0.0,
    "hdbscan_max_cluster_size": 0,
    "hdbscan_allow_single_cluster": False,
    "countvectorizer_stop_words": "french",
    "countvectorizer_ngram_range": (1, 2),
    "ctfidf_reduce_frequent_words": True,
    "ctfidf_bm25_weighting": False,
    "representation_model": ["MaximalMarginalRelevance"],
    "keybert_nr_repr_docs": 5,
    "keybert_nr_candidate_words": 40,
    "keybert_top_n_words": 20,
    "mmr_diversity": 0.2,
    "mmr_top_n_words": 10,
    "data_language": "English",
}


# BERTopic options for Streamlit UI
def bertopic_options():
    return {
        "bertopic_nr_topics": st.number_input(
            "nr_topics",
            min_value=0,
            value=DEFAULT_PARAMETERS["bertopic_nr_topics"],
            key="bertopic_nr_topics",
        ),
        "bertopic_top_n_words": st.number_input(
            "top_n_words",
            min_value=1,
            value=DEFAULT_PARAMETERS["bertopic_top_n_words"],
            key="bertopic_top_n_words",
        ),
    }


# C-TF-IDF options for Streamlit UI
def ctfidf_options():
    return {
        "ctfidf_reduce_frequent_words": st.checkbox(
            "reduce_frequent_words",
            value=DEFAULT_PARAMETERS["ctfidf_reduce_frequent_words"],
            key="ctfidf_reduce_frequent_words",
        ),
        "ctfidf_bm25_weighting": st.checkbox(
            "bm25_weighting",
            value=DEFAULT_PARAMETERS["ctfidf_bm25_weighting"],
            key="ctfidf_bm25_weighting",
        ),
    }


# Representation model options for Streamlit UI
def representation_model_options():
    options = {}
    available_models = ["KeyBERTInspired", "MaximalMarginalRelevance", "OpenAI"]

    # Model-specific parameters (always visible for ALL models)
    for model in available_models:
        st.write(f"### {model} Parameters")
        if model == "KeyBERTInspired":
            options[f"{model}_nr_repr_docs"] = st.number_input(
                f"{model}: Number of representative documents",
                min_value=1,
                value=5,
                key=f"{model}_nr_repr_docs",
            )
            options[f"{model}_nr_candidate_words"] = st.number_input(
                f"{model}: Number of candidate words",
                min_value=1,
                value=40,
                key=f"{model}_nr_candidate_words",
            )
            options[f"{model}_top_n_words"] = st.number_input(
                f"{model}: Top N words",
                min_value=1,
                value=20,
                key=f"{model}_top_n_words",
            )
        elif model == "MaximalMarginalRelevance":
            options[f"{model}_diversity"] = st.slider(
                f"{model}: Diversity",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                key=f"{model}_diversity",
            )
            options[f"{model}_top_n_words"] = st.number_input(
                f"{model}: Top N words",
                min_value=1,
                value=10,
                key=f"{model}_top_n_words",
            )
        elif model == "OpenAI":
            options[f"{model}_nr_docs"] = st.number_input(
                f"{model}: Number of documents",
                min_value=1,
                value=5,
                key=f"{model}_nr_docs",
            )
            options[f"{model}_language"] = st.selectbox(
                f"{model}: Data Language",
                options=["Français", "English"],
                key=f"{model}_language",
            )

        st.divider()  # Add a divider after each model's parameters, regardless of selection

    st.write("Select representation models:")
    selected_models = st.multiselect(
        "Models",
        options=available_models,
        default=DEFAULT_PARAMETERS["representation_model"],
        key="representation_model",
    )

    # if selected, move OpenAI model to the end of the list
    if "OpenAI" in selected_models:
        selected_models = [model for model in selected_models if model != "OpenAI"] + [
            "OpenAI"
        ]

    options["representation_model"] = selected_models
    return options


def _make_dynamic_topics_split(df, nr_bins):
    """
    Split docs into nr_bins and generate a llm_utils timestamp label into a new column
    """
    df = df.sort_values(TIMESTAMP_COLUMN, ascending=False)
    split_df = np.array_split(df, nr_bins)
    for split in split_df:
        split[GROUPED_TIMESTAMP_COLUMN] = split[TIMESTAMP_COLUMN].max()
    return pd.concat(split_df)


@st.cache_data
def compute_topics_over_time(
    _topic_model,
    df,
    nr_bins,
):
    df = _make_dynamic_topics_split(df, nr_bins)
    res = _topic_model.topics_over_time(
        df[TEXT_COLUMN],
        df[GROUPED_TIMESTAMP_COLUMN],
        global_tuning=False,
    )
    return res


def print_docs_for_specific_topic(df, topics, topic_number):
    """
    Print documents for a specific topic
    """
    columns_list = [
        col
        for col in [
            TITLE_COLUMN,
            TEXT_COLUMN,
            URL_COLUMN,
            TIMESTAMP_COLUMN,
            CITATION_COUNT_COL,
        ]
        if col in df.keys()
    ]
    df = df.loc[pd.Series(topics) == topic_number][columns_list]
    for _, doc in df.iterrows():
        st.write(f"[{doc.title}]({doc.url})")


# TODO: Remove "put embeddings in cache" option since it's unadvised due to the large size of embeddings returned by embedding model (sentence and token embeddings)
# TODO: Make the parameters of different representation models appear and disappear based on what was selected in the multi-select box.
