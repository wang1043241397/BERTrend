import pandas as pd
import streamlit as st

from streamlit.runtime.uploaded_file_manager import UploadedFile

from bertrend.BERTopicModel import BERTopicModel
from bertrend.utils.data_loading import (
    TEXT_COLUMN,
    TIMESTAMP_COLUMN,
    clean_dataset,
    enhanced_split_df_by_paragraphs,
)
from bertrend_apps.exploration.curebot.app_utils import (
    chunk_df,
    concat_data_from_files,
    fit_bertopic,
    get_embeddings,
    update_topics_per_document,
    get_improved_topic_description,
)


def show() -> None:
    # Data uploading component
    upload_data()

    # Load data into dataframe and split dataframe
    if st.session_state.get("uploaded_files"):
        preprocess_data()

    # If data is loaded
    if "df" in st.session_state:
        # Show data
        st.write(st.session_state["df"])

        # Button to train model
        if st.button("Détecter les sujets"):
            with st.spinner("Détection des sujets..."):
                # Train model on split data
                train_model()
            st.success("Sujets détectés, voir l'onglet résultats.")

        # If topic model is trained, update df with topics
        if "topic_model" in st.session_state:
            st.session_state["df"] = update_topics_per_document(
                st.session_state["df"],
                st.session_state["df_split"],
                st.session_state["topics"],
            )

            st.session_state["topics_info"]["llm_description"] = (
                get_improved_topic_description(
                    st.session_state["df"], st.session_state["topics_info"]
                )
            )


def upload_data() -> None:
    """
    Data uploading component for Curebot format.
    Sets in sessions_state:
      - "uploaded_files": list of uploaded files
      - "atom_rss_url": URL of the ATOM/RSS feed
    """
    with st.expander("**Import des données de Curebot**", expanded=True):
        # ATOM / RSS input
        st.session_state["atom_rss_url"] = st.text_input(
            "URL du flux ATOM / RSS",
            key="curebot_rss_url",
            help="Saisir l'URL complète du flux Curebot à importer (par ex. https://api-a1.beta.curebot.io/v1/atom-feed/smartfolder/a5b14e159caa4cb5967f94e84640f602)",
        )

        # Excel files input
        st.session_state["uploaded_files"] = st.file_uploader(
            "Fichiers Excel (format Curebot .xlsx)",
            accept_multiple_files=True,
            help="Glisser/déposer dans cette zone les exports Curebot au format Excel",
        )


def preprocess_data() -> None:
    """
    Preprocess data from uploaded files.
    Sets in session_state:
      - "df": dataframe with all data
      - "df_split": dataframe with texts split by paragraphs
    """
    # st.session_state["df"] = parse_data_from_files(uploaded_files)
    st.session_state["df"] = concat_data_from_files(st.session_state["uploaded_files"])
    st.session_state["df_split"] = chunk_df(st.session_state["df"])


def train_model() -> None:
    """
    Train a BERTopic model based on provided data.
    Sets in session_state:
      - "embeddings": embeddings of the dataset
      - "topic_model": trained BERTopic model
      - "topics": topics extracted by the model
      - "topics_info": information about the topics
    """
    # Get dataset and embeddings
    dataset = st.session_state["df_split"][TEXT_COLUMN].tolist()
    st.session_state["embeddings"] = get_embeddings(dataset)

    # Convert tags to string
    st.session_state["df"]["Tags"] = st.session_state["df"]["Tags"].astype(str)

    # Get zeroshot_topic_list from tags
    zeroshot_topic_list = (
        st.session_state["df"]["Tags"]
        .fillna("")
        .str.findall(r"#\w+")
        .explode()
        .unique()
    )

    zeroshot_topic_list = [
        str(tag).replace("#", "").replace("_", " ")
        for tag in zeroshot_topic_list
        if tag
    ]

    # Train topic model
    bertopic, topics = fit_bertopic(
        dataset, st.session_state["embeddings"], zeroshot_topic_list
    )

    # Set session_state
    st.session_state["topic_model"] = bertopic
    st.session_state["topics"] = topics

    topic_info = bertopic.get_topic_info()
    st.session_state["topics_info"] = topic_info[
        topic_info["Topic"] != -1
    ]  # exclude -1 topic from topic list
