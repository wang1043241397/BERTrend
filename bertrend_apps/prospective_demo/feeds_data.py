#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import os
import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger

from bertrend import FEED_BASE_PATH
from bertrend.utils.data_loading import (
    load_data,
    TIMESTAMP_COLUMN,
    TITLE_COLUMN,
    URL_COLUMN,
    TEXT_COLUMN,
)


def display_data_status():
    if not st.session_state.user_feeds:
        return

    col1, col2 = st.columns(2)
    with col1:
        st.selectbox(
            "Sélection de la veille",
            options=sorted(st.session_state.user_feeds.keys()),
            key="id_data",
        )

    with col2:
        if "time_window" not in st.session_state:
            st.session_state.time_window = 7
        st.slider(
            "Fenêtre temporelle (jours)",
            min_value=1,
            max_value=60,
            step=1,
            key="time_window",
        )

    display_data_info_for_feed(st.session_state.id_data)


def display_data_info_for_feed(feed_id: str):
    all_files = get_all_files_for_feed(feed_id)
    df = get_all_data(files=all_files)
    df = df[
        [TITLE_COLUMN, URL_COLUMN, TEXT_COLUMN, TIMESTAMP_COLUMN]
    ]  # filter useful columns

    if df.empty:
        df_filtered = pd.DataFrame()
    else:
        cutoff_date = datetime.datetime.now() - datetime.timedelta(
            days=st.session_state.time_window
        )
        df_filtered = df[df[TIMESTAMP_COLUMN] >= cutoff_date]

    stats = {
        "ID": feed_id,
        "# Fichiers": len(all_files),
        "Date début": df[TIMESTAMP_COLUMN].min() if not df.empty else None,
        "Date fin": df[TIMESTAMP_COLUMN].max() if not df.empty else None,
        "Nombre d'articles": len(df),
        f"Nombre d'articles (derniers {st.session_state.time_window} jours)": len(
            df_filtered
        ),
    }

    st.dataframe(pd.DataFrame([stats]))

    st.write(f"#### Données des derniers {st.session_state.time_window} jours")
    st.dataframe(
        df_filtered,
        use_container_width=True,
        hide_index=True,
        column_config={"url": st.column_config.LinkColumn("url")},
    )


@st.cache_data
def get_all_data(files: list[Path]) -> pd.DataFrame:
    """Returns the data contained in the provided files as a single DataFrame."""
    if not files:
        return pd.DataFrame()
    dfs = [load_data(Path(f)) for f in files]
    new_df = pd.concat(dfs).drop_duplicates(
        subset=["title"], keep="first", inplace=False
    )
    return new_df


def get_all_files_for_feed(feed_id: str) -> list[Path]:
    """Returns the paths of all files associated to a feed for the current user."""
    feed_base_dir = st.session_state.user_feeds[feed_id]["data-feed"]["feed_dir_path"]
    list_all_files = list(
        Path(FEED_BASE_PATH, feed_base_dir).glob(
            f"*{st.session_state.user_feeds[feed_id]['data-feed'].get('id')}*.jsonl*"
        )
    )
    return list_all_files


def get_last_files(files: list[Path], time_window: int) -> list[Path] | None:
    """Returns the paths of all files associated to a feed for the current user in the last time window."""
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=time_window)
    matching_files = []
    for file in files:
        try:
            file_stat = file.stat()  # Get file stats only once
            print(file_stat)
            file_time = datetime.datetime.fromtimestamp(file_stat.st_mtime)
            if file_time >= cutoff_date:
                matching_files.append(file)
        except OSError as e:
            logger.warning(f"Error accessing file {file}: {e}")
            # Handle the error as needed (e.g., skip the file)


def get_first_file(files: list[Path]) -> Path | None:
    """Returns the first file associated to a feed for the current user."""
    if files:  # Check if any files were found
        first_file = min(files, key=os.path.getctime)
    else:
        first_file = None  # Or handle the case where no files are found appropriately.  Perhaps raise an exception.
    return first_file


def get_last_file(files: list[Path]) -> Path | None:
    """Returns the last file associated to a feed for the current user."""
    if files:  # Check if any files were found
        latest_file = max(files, key=os.path.getctime)
    else:
        latest_file = None  # Or handle the case where no files are found appropriately.  Perhaps raise an exception.
    return latest_file
