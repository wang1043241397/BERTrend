#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import pandas as pd
import streamlit as st

from bertrend.demos.demos_utils.icons import NEWSLETTER_ICON, TOPIC_ICON
from bertrend_apps.prospective_demo import (
    WEAK_SIGNALS,
    STRONG_SIGNALS,
    LLM_TOPIC_DESCRIPTION_COLUMN,
)

WEAK_SIGNAL_NB = 3
STRONG_SIGNAL_NB = 5


@st.fragment
def reporting():
    tab1, tab2 = st.tabs(
        [
            TOPIC_ICON + " Etape 1: Sélection des sujets à retenir",
            NEWSLETTER_ICON + " Etape 2",
        ]
    )
    with tab1:
        choose_topics()

    # generate_newsletter()


def choose_topics():
    st.subheader("Etape 1: Sélection des sujets à retenir")
    cols = st.columns(2)
    with cols[0]:
        st.write("#### :orange[Sujets émergents]")
        st.session_state.weak_topics_list = choose_from_df(
            st.session_state.signal_interpretations[WEAK_SIGNALS]
        )
        st.write(st.session_state.weak_topics_list)
    with cols[1]:
        st.write("#### :green[Sujets forts]")
        st.session_state.strong_topics_list = choose_from_df(
            st.session_state.signal_interpretations[STRONG_SIGNALS]
        )
        st.write(st.session_state.strong_topics_list)


def choose_from_df(df: pd.DataFrame):
    df["A retenir"] = True
    df["Sujet"] = df[LLM_TOPIC_DESCRIPTION_COLUMN].apply(lambda r: r["title"])
    df["Description"] = df[LLM_TOPIC_DESCRIPTION_COLUMN].apply(
        lambda r: r["description"]
    )
    columns = ["Topic", "A retenir", "Sujet", "Description"]
    pd.DataFrame(
        [
            {"command": "st.selectbox", "rating": 4, "is_widget": True},
            {"command": "st.balloons", "rating": 5, "is_widget": False},
            {"command": "st.time_input", "rating": 3, "is_widget": True},
        ]
    )
    edited_df = st.data_editor(df[columns], num_rows="dynamic", column_order=columns)
    selection = edited_df[edited_df["A retenir"] == True]["Topic"].tolist()
    return selection
