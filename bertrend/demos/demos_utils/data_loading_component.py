#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Literal

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from bertrend import DATA_PATH
from bertrend.demos.demos_utils.session_state_manager import SessionStateManager
from bertrend.parameters import MIN_CHARS_DEFAULT, SAMPLE_SIZE_DEFAULT
from bertrend.utils.data_loading import (
    find_compatible_files,
    load_and_preprocess_data,
    TEXT_COLUMN,
)

NO_DATASET_WARNING = "Please select at least one dataset to proceed."
FORMAT_ICONS = {
    "csv": "📊",
    "parquet": "📦",
    "json": "📜",
    "jsonl": "📜",
    "xlsx": "📊",
}


def _process_uploaded_files(
    files: List[UploadedFile],
    min_chars: int,
    split_by_paragraph=Literal["yes", "no", "enhanced"],
) -> List[pd.DataFrame]:
    """Read a list of Excel files and return a single dataframe containing the data"""
    dataframes = []
    with TemporaryDirectory() as tmpdir:
        for f in files:
            with open(tmpdir + "/" + f.name, "wb") as tmp_file:
                tmp_file.write(f.getvalue())
            if tmp_file is not None:
                df = load_and_preprocess_data(
                    Path(tmp_file.name),
                    st.session_state["language"],
                    min_chars,
                    split_by_paragraph,
                    embedding_model_name=SessionStateManager.get(
                        "embedding_model_name"
                    ),
                )
                dataframes.append(df)
        return dataframes


def _load_files(
    files: List[Path],
    min_chars: int,
    split_by_paragraph=Literal["yes", "no", "enhanced"],
) -> List[pd.DataFrame]:
    dfs = []
    for selected_file in files:
        file_path = DATA_PATH / selected_file
        df = load_and_preprocess_data(
            file_path,
            st.session_state["language"],
            min_chars,
            split_by_paragraph,
            embedding_model_name=SessionStateManager.get("embedding_model_name"),
        )
        dfs.append(df)
    return dfs


def display_data_loading_component():
    """
    Component for a streamlit app about topic modelling. It allows to choose data to load and preprocess data.
    The final dataframe is stored inside the Streamlit state variable "time_filtered_df"
    """
    # Find files in the current directory and subdirectories
    tab1, tab2 = st.tabs(["Data from local storage", "Data from server data"])
    compatible_extensions = FORMAT_ICONS.keys()

    with tab1:
        uploaded_files = st.file_uploader(
            label="Select dataset from local storage (.xlsx, .csv, .json, .jsonl, .parquet)",
            type=compatible_extensions,
            accept_multiple_files=True,
            help="Drag and drop files to be used as dataset in this area",
        )

    with tab2:
        selected_files = st.multiselect(
            label="Select one or more datasets from the server data",
            options=find_compatible_files(DATA_PATH, compatible_extensions),
            default=SessionStateManager.get("selected_files", []),
            key="selected_files",
            format_func=lambda x: FORMAT_ICONS[x.suffix.lstrip(".")] + " " + str(x),
        )

    if uploaded_files is None and not selected_files:
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
        split_by_paragraph = st.segmented_control(
            "Split text by paragraphs",
            key="split_by_paragraph",
            options=["no", "yes", "enhanced"],
            default="yes",
            selection_mode="single",
            help="""- No split: No splitting on the documents.
            
            - Split by paragraphs: Split documents into paragraphs.
            
            - Enhanced split: uses a more advanced but slower method for splitting that considers the embedding model's maximum input length.
            """,
        )
    # Load and preprocess each selected file, then concatenate them
    # Priority to local data if both are set
    if uploaded_files is not None:
        dfs = _process_uploaded_files(uploaded_files, min_chars, split_by_paragraph)
    elif selected_files:
        dfs = _load_files(selected_files, min_chars, split_by_paragraph)

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

        SessionStateManager.set("time_filtered_df", df_filtered)
        st.write(
            f"Number of documents in selected timeframe: {len(SessionStateManager.get_dataframe('time_filtered_df'))}"
        )
        st.dataframe(
            SessionStateManager.get_dataframe("time_filtered_df")[
                [TEXT_COLUMN, "timestamp"]
            ],
            use_container_width=True,
        )
