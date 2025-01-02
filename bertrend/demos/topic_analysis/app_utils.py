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
