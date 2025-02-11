#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import streamlit as st

from bertrend.demos.weak_signals.visualizations_utils import (
    display_signal_categories_df,
)
from bertrend_apps.prospective_demo import (
    LLM_TOPIC_DESCRIPTION_COLUMN,
    NOISE,
    WEAK_SIGNALS,
    STRONG_SIGNALS,
    get_model_interpretation_path,
)
from bertrend_apps.prospective_demo.dashboard_common import (
    choose_id_and_ts,
    get_df_topics,
)


def signal_analysis():
    st.write(
        "Ici mettre seulement les tableaux weak / strong + les liens vers les articles"
    )
    # ID and timestamp selection
    choose_id_and_ts()
    model_id = st.session_state.model_id
    reference_ts = st.session_state.reference_ts

    model_interpretation_path = get_model_interpretation_path(
        user_name=st.session_state.username,
        model_id=model_id,
        reference_ts=reference_ts,
    )

    # Display dataframes for weak_signals, strong, etc
    # Display data frames
    columns = [
        "Topic",
        LLM_TOPIC_DESCRIPTION_COLUMN,
        "Representation",
        "Latest_Popularity",
        "Docs_Count",
        "Paragraphs_Count",
        "Latest_Timestamp",
        "Documents",
        "Sources",
        "Source_Diversity",
    ]

    dfs_topics = get_df_topics(model_interpretation_path)
    display_signal_categories_df(
        dfs_topics[NOISE],
        dfs_topics[WEAK_SIGNALS],
        dfs_topics[STRONG_SIGNALS],
        reference_ts,
        columns=columns,
    )
