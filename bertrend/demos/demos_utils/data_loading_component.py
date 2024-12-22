#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import pandas as pd
import streamlit as st

from bertrend import DATA_PATH
from bertrend.demos.demos_utils.session_state_manager import SessionStateManager
from bertrend.parameters import MIN_CHARS_DEFAULT, SAMPLE_SIZE_DEFAULT
from bertrend.utils.data_loading import (
    find_compatible_files,
    load_and_preprocess_data,
    TEXT_COLUMN,
)

NO_DATASET_WARNING = "Please select at least one dataset to proceed."


def display_data_loading_component():
    # Find files in the current directory and subdirectories
    compatible_extensions = ["csv", "parquet", "json", "jsonl"]
    selected_files = st.multiselect(
        "Select one or more datasets",
        find_compatible_files(DATA_PATH, compatible_extensions),
        default=SessionStateManager.get("selected_files", []),
        key="selected_files",
    )

    if not selected_files:
        st.warning(NO_DATASET_WARNING)
        return

    # Display number input and checkbox for preprocessing options
    col1, col2 = st.columns(2)
    with col1:
        min_chars = st.number_input(
            "Minimum Characters",
            value=MIN_CHARS_DEFAULT,
            min_value=0,
            max_value=1000,
            key="min_chars",
        )
    with col2:
        split_by_paragraph = st.checkbox(
            "Split text by paragraphs", value=False, key="split_by_paragraph"
        )

    # Load and preprocess each selected file, then concatenate them
    dfs = []
    for selected_file, ext in selected_files:
        file_path = DATA_PATH / selected_file
        df = load_and_preprocess_data(
            (file_path, ext),
            st.session_state["language"],
            min_chars,
            split_by_paragraph,
        )
        dfs.append(df)

    if not dfs:
        st.warning(
            "No data available after preprocessing. Please check the selected files and preprocessing options."
        )
    else:
        df = pd.concat(dfs, ignore_index=True)

        # Deduplicate using all columns
        df = df.drop_duplicates()

        # Select timeframe
        min_date, max_date = df["timestamp"].dt.date.agg(["min", "max"])
        start_date, end_date = st.slider(
            "Select Timeframe",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            key="timeframe_slider",
        )

        # Filter and sample the DataFrame
        df_filtered = df[
            (df["timestamp"].dt.date >= start_date)
            & (df["timestamp"].dt.date <= end_date)
        ]
        df_filtered = df_filtered.sort_values(by="timestamp").reset_index(drop=True)

        sample_size = st.number_input(
            "Sample Size",
            value=SAMPLE_SIZE_DEFAULT or len(df_filtered),
            min_value=1,
            max_value=len(df_filtered),
            key="sample_size",
        )
        if sample_size < len(df_filtered):
            df_filtered = df_filtered.sample(n=sample_size, random_state=42)

        df_filtered = df_filtered.sort_values(by="timestamp").reset_index(drop=True)

        SessionStateManager.set("timefiltered_df", df_filtered)
        st.write(
            f"Number of documents in selected timeframe: {len(SessionStateManager.get_dataframe('timefiltered_df'))}"
        )
        st.dataframe(
            SessionStateManager.get_dataframe("timefiltered_df")[
                [TEXT_COLUMN, "timestamp"]
            ],
            use_container_width=True,
        )
