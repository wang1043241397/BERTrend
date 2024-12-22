#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pickle
import shutil

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger

from bertrend import (
    ZEROSHOT_TOPICS_DATA_DIR,
    CACHE_PATH,
)
from bertrend.BERTrend import BERTrend
from bertrend.demos.demos_utils.data_loading_component import (
    display_data_loading_component,
)
from bertrend.demos.demos_utils.parameters_component import (
    display_bertopic_hyperparameters,
)
from bertrend.services.embedding_service import EmbeddingService
from bertrend.topic_model import TopicModel
from bertrend.demos.weak_signals.messages import (
    MODEL_MERGING_COMPLETE_MESSAGE,
    NO_CACHE_WARNING,
    CACHE_PURGED_MESSAGE,
    MODELS_RESTORED_MESSAGE,
    EMBEDDINGS_CALCULATED_MESSAGE,
    NO_DATA_WARNING,
    MODEL_TRAINING_COMPLETE_MESSAGE,
    STATE_SAVED_MESSAGE,
    STATE_RESTORED_MESSAGE,
    MODELS_SAVED_MESSAGE,
    NO_MODELS_WARNING,
)
from bertrend.trend_analysis.weak_signals import detect_weak_signals_zeroshot

from bertrend.utils.data_loading import (
    group_by_days,
    TEXT_COLUMN,
)
from bertrend.parameters import *
from bertrend.demos.demos_utils.session_state_manager import SessionStateManager
from bertrend.trend_analysis.visualizations import (
    plot_size_outliers,
    plot_num_topics,
)
from bertrend.demos.weak_signals.visualizations_utils import (
    PLOTLY_BUTTON_SAVE_CONFIG,
    display_sankey_diagram,
    display_topics_per_timestamp,
    display_newly_emerged_topics,
    display_popularity_evolution,
    save_signal_evolution,
    display_signal_analysis,
    retrieve_topic_counts,
)

# UI Settings
PAGE_TITLE = "BERTrend - Trend Analysis"
LAYOUT = "wide"


def save_state():
    state_file = CACHE_PATH / STATE_FILE
    embeddings_file = CACHE_PATH / EMBEDDINGS_FILE

    # Save the selected files (list of filenames)
    selected_files = SessionStateManager.get("selected_files", [])

    state = SessionStateManager.get_multiple(
        "selected_files",
        "min_chars",
        "split_by_paragraph",
        "timeframe_slider",
        "language",
        "embedding_model_name",
        "embedding_model",
        "sample_size",
        "min_similarity",
        "zeroshot_min_similarity",
        "embedding_dtype",
        "data_embedded",
    )

    state["selected_files"] = selected_files

    with open(state_file, "wb") as f:
        pickle.dump(state, f)

    np.save(embeddings_file, SessionStateManager.get_embeddings())
    st.success(STATE_SAVED_MESSAGE)


def restore_state():
    state_file = CACHE_PATH / STATE_FILE
    embeddings_file = CACHE_PATH / EMBEDDINGS_FILE

    if state_file.exists() and embeddings_file.exists():
        with open(state_file, "rb") as f:
            state = pickle.load(f)

        # Restore the selected files
        selected_files = state.get("selected_files", [])
        SessionStateManager.set("selected_files", selected_files)

        # Restore other states
        SessionStateManager.set_multiple(**state)
        SessionStateManager.set("embeddings", np.load(embeddings_file))
        st.success(STATE_RESTORED_MESSAGE)

        # Update the multiselect widget with restored selected files
        st.session_state["selected_files"] = selected_files
    else:
        st.warning("No saved state found.")


def purge_cache():
    if CACHE_PATH.exists():
        shutil.rmtree(CACHE_PATH)
        st.success(CACHE_PURGED_MESSAGE)
    else:
        st.warning(NO_CACHE_WARNING)


def main():
    st.set_page_config(
        page_title=PAGE_TITLE, layout=LAYOUT, initial_sidebar_state="expanded"
    )

    st.title(":part_alternation_mark: " + PAGE_TITLE)

    # Set the main flags
    SessionStateManager.get_or_set("data_embedded", False)
    SessionStateManager.get_or_set("popularity_computed", False)

    # Sidebar
    with st.sidebar:
        st.header("Settings and Controls")

        # State Management
        st.subheader("State Management")

        if st.button("Restore Previous Run", use_container_width=True):
            restore_state()
            try:
                SessionStateManager.set("bertrend", BERTrend.restore_models())
                st.success(MODELS_RESTORED_MESSAGE)
            except Exception as e:
                st.warning(NO_MODELS_WARNING)

        if st.button("Purge Cache", use_container_width=True):
            purge_cache()

        if st.button("Clear session state", use_container_width=True):
            SessionStateManager.clear()

        # BERTopic Hyperparameters
        st.subheader("BERTopic Hyperparameters")
        display_bertopic_hyperparameters()

    # Main content
    tab1, tab2, tab3 = st.tabs(["Data Loading", "Model Training", "Results Analysis"])

    with tab1:
        st.header("Data Loading and Preprocessing")

        display_data_loading_component()

        if "timefiltered_df" in st.session_state:

            # Embed documents
            if st.button("Embed Documents"):
                with st.spinner("Embedding documents..."):
                    embedding_dtype = SessionStateManager.get("embedding_dtype")
                    embedding_model_name = SessionStateManager.get(
                        "embedding_model_name"
                    )
                    embedding_service = EmbeddingService()

                    texts = SessionStateManager.get_dataframe("timefiltered_df")[
                        TEXT_COLUMN
                    ].tolist()

                    try:
                        embedding_model, embeddings = embedding_service.embed_documents(
                            texts=texts,
                            embedding_model_name=embedding_model_name,
                            embedding_dtype=embedding_dtype,
                        )

                        SessionStateManager.set("embedding_model", embedding_model)
                        SessionStateManager.set("embeddings", embeddings)
                        SessionStateManager.set("data_embedded", True)

                        st.success(EMBEDDINGS_CALCULATED_MESSAGE)
                        save_state()
                    except Exception as e:
                        st.error(
                            f"An error occurred while embedding documents: {str(e)}"
                        )

    with tab2:
        st.header("Model Training")

        # Select granularity
        st.number_input(
            "Select Granularity",
            value=DEFAULT_GRANULARITY,
            min_value=1,
            max_value=30,
            key="granularity_select",
            help="Number of days to split the data by",
        )

        # Show documents per grouped timestamp
        with st.expander("Documents per Timestamp", expanded=True):
            grouped_data = group_by_days(
                SessionStateManager.get_dataframe("timefiltered_df"),
                day_granularity=SessionStateManager.get("granularity_select"),
            )
            non_empty_timestamps = [
                timestamp
                for timestamp, group in grouped_data.items()
                if not group.empty
            ]
            if non_empty_timestamps:
                selected_timestamp = st.select_slider(
                    "Select Timestamp",
                    options=non_empty_timestamps,
                    key="timestamp_slider",
                )
                selected_docs = grouped_data[selected_timestamp]
                st.dataframe(
                    selected_docs[
                        ["timestamp", TEXT_COLUMN, "document_id", "source", "url"]
                    ],
                    use_container_width=True,
                )
            else:
                st.warning(NO_DATA_WARNING)

        if not SessionStateManager.get("data_embedded", False):
            st.warning("Please embed data before proceeding to model training.")
            st.stop()
        else:
            # Zero-shot topic definition
            zeroshot_topic_list = st.text_input(
                "Enter zero-shot topics (separated by /)", value=""
            )
            zeroshot_topic_list = [
                topic.strip()
                for topic in zeroshot_topic_list.split("/")
                if topic.strip()
            ]

            if st.button("Train Models"):
                with st.spinner("Training models..."):
                    # FIXME: called twice (see above)
                    grouped_data = group_by_days(
                        SessionStateManager.get_dataframe("timefiltered_df"),
                        day_granularity=SessionStateManager.get("granularity_select"),
                    )

                    logger.debug(SessionStateManager.get("language"))

                    # Initialize topic model
                    topic_model = TopicModel(
                        umap_n_components=SessionStateManager.get("umap_n_components"),
                        umap_n_neighbors=SessionStateManager.get("umap_n_neighbors"),
                        hdbscan_min_cluster_size=SessionStateManager.get(
                            "hdbscan_min_cluster_size"
                        ),
                        hdbscan_min_samples=SessionStateManager.get(
                            "hdbscan_min_samples"
                        ),
                        hdbscan_cluster_selection_method=SessionStateManager.get(
                            "hdbscan_cluster_selection_method"
                        ),
                        vectorizer_ngram_range=SessionStateManager.get(
                            "vectorizer_ngram_range"
                        ),
                        min_df=SessionStateManager.get("min_df"),
                        top_n_words=SessionStateManager.get("top_n_words"),
                        language=SessionStateManager.get("language"),
                    )

                    # Created BERTrend object
                    bertrend = BERTrend(
                        topic_model=topic_model,
                        zeroshot_topic_list=zeroshot_topic_list,
                        zeroshot_min_similarity=SessionStateManager.get(
                            "zeroshot_min_similarity"
                        ),
                    )
                    # Train topic models on data
                    bertrend.train_topic_models(
                        grouped_data=grouped_data,
                        embedding_model=SessionStateManager.get("embedding_model"),
                        embeddings=SessionStateManager.get_embeddings(),
                    )
                    st.success(MODEL_TRAINING_COMPLETE_MESSAGE)

                    # Save trained models
                    bertrend.save_models()
                    st.success(MODELS_SAVED_MESSAGE)

                    # Store bertrend object
                    SessionStateManager.set("bertrend", bertrend)

            if (
                "bertrend" not in st.session_state
                or not SessionStateManager.get("bertrend")._is_fitted
            ):
                st.stop()
            else:
                if st.button("Merge Models"):
                    with st.spinner("Merging models..."):
                        bertrend = SessionStateManager.get("bertrend")
                        bertrend.merge_models(
                            min_similarity=SessionStateManager.get("min_similarity"),
                        )

                        bertrend.calculate_signal_popularity(
                            granularity=SessionStateManager.get("granularity_select"),
                        )

                        SessionStateManager.set("popularity_computed", True)

                    st.success(MODEL_MERGING_COMPLETE_MESSAGE)

    with tab3:
        st.header("Results Analysis")

        if not SessionStateManager.get("data_embedded", False):
            st.warning(
                "Please embed data and train models before proceeding to analysis."
            )
            st.stop()

        elif not SessionStateManager.get("bertrend")._is_fitted:
            st.warning("Please train models before proceeding to analysis.")
            st.stop()

        else:
            topic_models = SessionStateManager.get("bertrend").topic_models
            st.subheader("Topic Overview")
            # Number of Topics Detected for each topic model
            st.plotly_chart(
                plot_num_topics(topic_models),
                config=PLOTLY_BUTTON_SAVE_CONFIG,
                use_container_width=True,
            )
            # Size of Outlier Topic for each topic model
            st.plotly_chart(
                plot_size_outliers(topic_models),
                config=PLOTLY_BUTTON_SAVE_CONFIG,
                use_container_width=True,
            )

            display_topics_per_timestamp(topic_models)

            # Display zeroshot signal trend
            if zeroshot_topic_list:
                st.subheader("Zero-shot Weak Signal Trends")
                weak_signal_trends = detect_weak_signals_zeroshot(
                    topic_models,
                    zeroshot_topic_list,
                    SessionStateManager.get("granularity_select"),
                )
                with st.expander("Zero-shot Weak Signal Trends", expanded=False):
                    fig_trend = go.Figure()
                    for topic, weak_signal_trend in weak_signal_trends.items():
                        timestamps = list(weak_signal_trend.keys())
                        popularity = [
                            weak_signal_trend[timestamp]["Document_Count"]
                            for timestamp in timestamps
                        ]
                        hovertext = [
                            f"Topic: {topic}<br>Timestamp: {timestamp}<br>Popularity: {weak_signal_trend[timestamp]['Document_Count']}<br>Representation: {weak_signal_trend[timestamp]['Representation']}"
                            for timestamp in timestamps
                        ]
                        fig_trend.add_trace(
                            go.Scatter(
                                x=timestamps,
                                y=popularity,
                                mode="lines+markers",
                                name=topic,
                                hovertext=hovertext,
                                hoverinfo="text",
                            )
                        )
                    fig_trend.update_layout(
                        title="Popularity of Zero-Shot Topics",
                        xaxis_title="Timestamp",
                        yaxis_title="Popularity",
                    )
                    st.plotly_chart(
                        fig_trend,
                        config=PLOTLY_BUTTON_SAVE_CONFIG,
                        use_container_width=True,
                    )

                    # Display the dataframe with zeroshot topics information
                    zeroshot_topics_data = [
                        {
                            "Topic": topic,
                            "Timestamp": timestamp,
                            "Representation": data["Representation"],
                            "Representative_Docs": data["Representative_Docs"],
                            "Count": data["Count"],
                            "Document_Count": data["Document_Count"],
                        }
                        for topic, weak_signal_trend in weak_signal_trends.items()
                        for timestamp, data in weak_signal_trend.items()
                    ]
                    zeroshot_topics_df = pd.DataFrame(zeroshot_topics_data)
                    st.dataframe(zeroshot_topics_df, use_container_width=True)

                    # Save the zeroshot topics data to a JSON file
                    json_file_path = ZEROSHOT_TOPICS_DATA_DIR
                    json_file_path.mkdir(parents=True, exist_ok=True)

                    zeroshot_topics_df.to_json(
                        json_file_path / ZEROSHOT_TOPICS_DATA_FILE,
                        orient="records",
                        date_format="iso",
                        indent=4,
                    )
                    st.success(f"Zeroshot topics data saved to {json_file_path}")

            if not SessionStateManager.get("popularity_computed", False):
                st.warning("Please merge models to view additional analyses.")
                st.stop()

            else:
                # Display merged signal trend
                st.subheader("Topic Size Evolution")
                st.dataframe(
                    SessionStateManager.get("bertrend").all_merge_histories_df[
                        [
                            "Timestamp",
                            "Topic1",
                            "Topic2",
                            "Representation1",
                            "Representation2",
                            "Document_Count1",
                            "Document_Count2",
                        ]
                    ]
                )

                # Display topic popularity evolution
                with st.expander("Topic Popularity Evolution", expanded=True):
                    display_popularity_evolution()
                    # Save Signal Evolution Data to investigate later on in a separate notebook
                    save_signal_evolution()

                # Analyze signal
                st.subheader("Signal Analysis")
                topic_number = st.number_input(
                    "Enter a topic number to take a closer look:", min_value=0, step=1
                )
                if st.button("Analyze signal"):
                    try:
                        display_signal_analysis(topic_number)
                    except Exception as e:
                        st.error(f"Error while trying to generate signal summary: {e}")

                # Create the Sankey Diagram
                st.subheader("Topic Evolution")
                display_sankey_diagram(
                    SessionStateManager.get("bertrend").all_merge_histories_df
                )

                # Newly emerged topics
                if SessionStateManager.get("bertrend").all_new_topics_df is not None:
                    st.subheader("Newly Emerged Topics")
                    display_newly_emerged_topics(
                        SessionStateManager.get("bertrend").all_new_topics_df
                    )

                if st.button("Retrieve Topic Counts"):
                    with st.spinner("Retrieving topic counts..."):
                        # Number of topics per individual topic model
                        retrieve_topic_counts(topic_models)
                        st.success(
                            f"Topic counts for individual and cumulative merged models saved to {json_file_path}"
                        )


if __name__ == "__main__":
    main()
