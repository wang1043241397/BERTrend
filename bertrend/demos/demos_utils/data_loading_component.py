#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from bertrend import DATA_PATH
from bertrend.demos.demos_utils.icons import (
    WARNING_ICON,
    JSON_ICON,
    CSV_ICON,
    PARQUET_ICON,
    XLSX_ICON,
    CLIENT_STORAGE_ICON,
    SERVER_STORAGE_ICON,
)
from bertrend.demos.demos_utils.messages import NO_DATA_AFTER_PREPROCESSING_MESSAGE
from bertrend.demos.demos_utils.state_utils import (
    save_widget_state,
    register_widget,
    SessionStateManager,
)
from bertrend.parameters import MIN_CHARS_DEFAULT, SAMPLE_SIZE_DEFAULT
from bertrend.utils.data_loading import (
    find_compatible_files,
    TEXT_COLUMN,
    load_data,
    split_data,
    TIMESTAMP_COLUMN,
)

NO_DATASET_WARNING = "Please select at least one dataset to proceed."
FORMAT_ICONS = {
    "xlsx": XLSX_ICON,
    "csv": CSV_ICON,
    "parquet": PARQUET_ICON,
    "json": JSON_ICON,
    "jsonl": JSON_ICON,
    "jsonlines": JSON_ICON,
    "jsonl.gz": JSON_ICON,
}


def _process_uploaded_files(
    files: List[UploadedFile],
) -> List[pd.DataFrame]:
    """Read a list of uploaded files and return a list of dataframes containing the associated data"""
    dataframes = []
    with TemporaryDirectory() as tmpdir:
        for f in files:
            with open(tmpdir + "/" + f.name, "wb") as tmp_file:
                tmp_file.write(f.getvalue())
            if tmp_file is not None:
                df = load_data(
                    Path(tmp_file.name),
                    st.session_state["language"],
                )
                if df is not None:
                    dataframes.append(df)
        return dataframes


def _load_files(
    files: List[Path],
) -> List[pd.DataFrame]:
    """Read a list of files from storage and return a list of dataframes containing the associated data"""
    dfs = []
    for selected_file in files:
        file_path = DATA_PATH / selected_file
        df = load_data(
            file_path,
            st.session_state["language"],
        )
        if df is not None:
            dfs.append(df)
    return dfs


# TODO: if loaded data column names do not match our default values, show a popup for column data mapping
def display_data_loading_component():
    """
    Component for a streamlit app about topic modelling. It allows to choose data to load and preprocess data.
    Preprocessing of data includes:
    - concatenation of data from different files
    - adding potentially missing columns to make all datasets homogeneous
    - removal of duplicates
    - splitting the dataset by paragraphs to avoid too long textes
    - filtering based on timestamp range
    - filtering based on a minimum number of characters

    The initial dataframe (one line per document) is stored after filtering of bad data inside a Streamlit state
    variable "initial_df". After split by paragraph, it is stored in "df_split". The final dataframe (possibly
    split by paragraph from initial documents and in all cases filtered by dates) is stored inside the Streamlit
    state variable "time_filtered_df".
    """
    # Find files in the current directory and subdirectories
    tab1, tab2 = st.tabs(
        [
            CLIENT_STORAGE_ICON + " Data from local storage",
            SERVER_STORAGE_ICON + " Data from server data",
        ]
    )
    compatible_extensions = FORMAT_ICONS.keys()

    with tab1:
        st.file_uploader(
            label="Select dataset from local storage (.xlsx, .csv, .json, .jsonl, .parquet)",
            type=compatible_extensions,
            accept_multiple_files=True,
            help="Drag and drop files to be used as dataset in this area",
            on_change=save_widget_state,
            key="uploaded_files",
        )

    with tab2:
        register_widget("selected_files")
        st.multiselect(
            label="Select one or more datasets from the server data",
            options=find_compatible_files(DATA_PATH, compatible_extensions),
            default=SessionStateManager.get("selected_files", []),
            key="selected_files",
            format_func=lambda x: FORMAT_ICONS[x.suffix.lstrip(".")] + " " + str(x),
            on_change=save_widget_state,
        )

    if not SessionStateManager.get(
        "uploaded_files", []
    ) and not SessionStateManager.get("selected_files"):
        st.warning(NO_DATASET_WARNING, icon=WARNING_ICON)
        st.stop()

    # Display number input and checkbox for preprocessing options
    col1, col2 = st.columns(2)
    with col1:
        register_widget("min_chars")
        st.number_input(
            "Minimum Characters",
            value=MIN_CHARS_DEFAULT,
            min_value=0,
            max_value=1000,
            key="min_chars",
            on_change=save_widget_state,
        )
    with col2:
        register_widget("split_by_paragraph")
        SessionStateManager.get_or_set("split_by_paragraph", "yes")
        st.segmented_control(
            "Split text by paragraphs",
            key="split_by_paragraph",
            options=["no", "yes", "enhanced"],
            selection_mode="single",
            help="'No split': No splitting on the documents ; 'Split by paragraphs': Split documents into paragraphs ; "
            "'Enhanced split': uses a more advanced but slower method for splitting that considers the embedding "
            "model's maximum input length.",
            on_change=save_widget_state,
        )
    # Load and preprocess each selected file, then concatenate them
    # Priority to local data if both are set
    dfs = None
    if SessionStateManager.get("uploaded_files"):
        dfs = _process_uploaded_files(
            SessionStateManager.get("uploaded_files"),
        )
    elif SessionStateManager.get("selected_files"):
        dfs = _load_files(
            SessionStateManager.get("selected_files"),
        )

    if not dfs:
        st.warning(
            NO_DATA_AFTER_PREPROCESSING_MESSAGE,
            icon=WARNING_ICON,
        )
    else:
        df = pd.concat(dfs, ignore_index=True)

        # Save state of initial DF (before split and data selection)
        st.session_state["initial_df"] = df.copy()

        df = split_data(
            df,
            SessionStateManager.get("min_chars"),
            SessionStateManager.get("split_by_paragraph"),
            embedding_model_name=SessionStateManager.get("embedding_model_name"),
        )

        # Deduplicate using all columns
        df = df.drop_duplicates()

        # Save state of split dataframe (before time-based filtering)
        st.session_state["split_df"] = df.copy()

        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            # Select timeframe
            min_date, max_date = df[TIMESTAMP_COLUMN].dt.date.agg(["min", "max"])
            register_widget("timeframe_slider")
            start_date, end_date = st.slider(
                "Select Timeframe",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                key="timeframe_slider",
                on_change=save_widget_state,
            )

            # Filter and sample the DataFrame
            df_filtered = df[
                (df[TIMESTAMP_COLUMN].dt.date >= start_date)
                & (df[TIMESTAMP_COLUMN].dt.date <= end_date)
            ]
            df_filtered = df_filtered.sort_values(by=TIMESTAMP_COLUMN).reset_index(
                drop=True
            )

        with col2:
            register_widget("sample_size")
            sample_size = st.number_input(
                "Sample Size",
                value=SAMPLE_SIZE_DEFAULT or len(df_filtered),
                min_value=1,
                max_value=len(df_filtered),
                key="sample_size",
                on_change=save_widget_state,
            )

        if sample_size < len(df_filtered):
            df_filtered = df_filtered.sample(n=sample_size, random_state=42)

        df_filtered = df_filtered.sort_values(by=TIMESTAMP_COLUMN).reset_index(
            drop=True
        )

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
