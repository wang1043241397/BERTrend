#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path

import pandas as pd
import streamlit as st

from bertrend.demos.demos_utils.icons import WARNING_ICON
from bertrend.demos.weak_signals.visualizations_utils import (
    display_signal_categories_df,
)
from bertrend_apps.prospective_demo import (
    INTERPRETATION_PATH,
    get_user_models_path,
    WEAK_SIGNALS,
    STRONG_SIGNALS,
    NOISE,
    LLM_TOPIC_DESCRIPTION_COLUMN,
    get_model_interpretation_path,
)
from bertrend_apps.prospective_demo.dashboard_common import (
    choose_id_and_ts,
    get_df_topics,
)


@st.fragment()
def dashboard_analysis():
    """Dashboard to analyze information monitoring results"""
    st.session_state.signal_interpretations = {}
    choose_id_and_ts()

    # LLM-based interpretation
    model_id = st.session_state.model_id
    reference_ts = st.session_state.reference_ts

    model_interpretation_path = get_model_interpretation_path(
        user_name=st.session_state.username,
        model_id=model_id,
        reference_ts=reference_ts,
    )

    # Detailed analysis
    st.subheader("Analyse détaillée par sujet")
    dfs_topics = get_df_topics(model_interpretation_path)
    display_detailed_analysis(model_id, model_interpretation_path, dfs_topics)


@st.fragment()
def display_detailed_analysis(
    model_id: str, model_interpretation_path: Path, dfs_topics: dict[str, pd.DataFrame]
):
    # Retrieve previously computed interpretation
    interpretations = {}
    for df_id, df in dfs_topics.items():
        if not df.empty:
            interpretation_file_path = (
                model_interpretation_path / f"{df_id}_interpretation.jsonl"
            )
            interpretations[df_id] = (
                pd.merge(
                    pd.read_json(interpretation_file_path, lines=True),
                    df,
                    how="left",
                    left_on="topic",
                    right_on="Topic",
                )
                if interpretation_file_path.exists()
                else {}
            )

    signal_topics = {WEAK_SIGNALS: [], STRONG_SIGNALS: []}
    if WEAK_SIGNALS in interpretations:
        signal_topics[WEAK_SIGNALS] = list(interpretations[WEAK_SIGNALS]["topic"])
    if STRONG_SIGNALS in interpretations:
        signal_topics[STRONG_SIGNALS] = list(interpretations[STRONG_SIGNALS]["topic"])
    signal_list = signal_topics[WEAK_SIGNALS] + signal_topics[STRONG_SIGNALS]
    selected_signal = st.selectbox(
        label="Sélection du sujet",
        label_visibility="hidden",
        options=signal_list,
        format_func=lambda signal_id: f"[Sujet {'émergent' if signal_id in signal_topics[WEAK_SIGNALS] else 'fort'} "
        f"{signal_id}]: {get_row(signal_id, interpretations[WEAK_SIGNALS] if signal_id in signal_topics[WEAK_SIGNALS] else interpretations[STRONG_SIGNALS])[LLM_TOPIC_DESCRIPTION_COLUMN]['title']}",
    )
    # Summary of the topic
    desc = get_row(
        selected_signal,
        (
            interpretations[WEAK_SIGNALS]
            if selected_signal in signal_topics[WEAK_SIGNALS]
            else interpretations[STRONG_SIGNALS]
        ),
    )
    if selected_signal in list(signal_topics[WEAK_SIGNALS]):
        color = "orange"
    else:
        color = "green"
    st.subheader(f":{color}[**{desc[LLM_TOPIC_DESCRIPTION_COLUMN]['title']}**]")
    st.write(desc[LLM_TOPIC_DESCRIPTION_COLUMN]["description"])
    # Detailed description
    st.html(desc["analysis"])

    st.session_state.signal_interpretations[model_id] = interpretations


def get_row(signal_id: int, df: pd.DataFrame) -> str:
    filtered_df = df[df["topic"] == signal_id]
    if not filtered_df.empty:
        return filtered_df.iloc[0]  # Return the Series (row)
    else:
        st.warning(f"No data found for signal ID: {signal_id}")
