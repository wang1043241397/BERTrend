import json
from bertopic import BERTopic
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import torch

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from tempfile import TemporaryDirectory
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
from urllib.parse import urlparse

from bertrend.BERTopicModel import BERTopicModel
from bertrend.llm_utils.openai_client import OpenAI_Client
from bertrend_apps.data_provider.curebot_provider import CurebotProvider
from bertrend.utils.data_loading import (
    load_data,
    TIMESTAMP_COLUMN,
    enhanced_split_df_by_paragraphs,
    clean_dataset,
)
from bertrend_apps.exploration.curebot.prompts import TOPIC_DESCRIPTION_SYSTEM_PROMPT

URL_COLUMN = "url"
TEXT_COLUMN = "text"
TITLE_COLUMN = "Titre de la ressource"

MIN_TEXT_LENGTH = 150
EMBEDDING_MODEL_NAME = "dangvantuan/french-document-embedding"
EMBEDDING_MODEL = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    model_kwargs={"torch_dtype": torch.float16},
    trust_remote_code=True,
)
TOP_N_WORDS = 5

PLOTLY_BUTTON_SAVE_CONFIG = {
    "toImageButtonOptions": {
        "format": "svg",
        # 'height': 500,
        # 'width': 1500,
        "scale": 1,
    }
}


@st.cache_data
def concat_data_from_files(files: list[UploadedFile]) -> pd.DataFrame:
    df_list = []
    for file in files:
        df_list.append(pd.read_excel(file))
    df = pd.concat(df_list, ignore_index=True)
    df = df.drop_duplicates(subset=[TITLE_COLUMN, TEXT_COLUMN]).reset_index(drop=True)
    return df


@st.cache_data
def chunk_df(
    df: pd.DataFrame, chunk_size: int = 100, overlap: int = 20
) -> pd.DataFrame:
    """
    Split df texts into overlapping chunks while preserving other columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing text and other columns
    chunk_size : int
        Number of words in each chunk
    overlap : int
        Number of words to overlap between chunks

    Returns:
    --------
    pandas.DataFrame
        DataFrame with chunked texts, preserving other column values
    """

    # # Apply chunking to each row
    # chunked_data = []
    # for _, row in df.iterrows():
    #     chunked_data.extend(split_text_to_chunks(row, chunk_size, overlap))

    # # Create new DataFrame from chunked data
    # df_split = (
    #     pd.DataFrame(chunked_data)
    #     .sort_values(by=TIMESTAMP_COLUMN)
    #     .reset_index(drop=True)
    # )

    return df.copy()


def split_text_to_chunks(row: pd.Series, chunk_size: int, overlap: int):
    # Split text into words
    words = row[TEXT_COLUMN].split()

    # If text is shorter than chunk size, return as is
    if len(words) <= chunk_size:
        new_row = row.copy()
        new_row[TEXT_COLUMN] = " ".join(words)
        return [new_row]

    # Create chunks with overlap
    chunks = []
    for start in range(0, len(words), chunk_size - overlap):
        chunk_words = words[start : start + chunk_size]
        new_row = row.copy()
        new_row[TEXT_COLUMN] = " ".join(chunk_words)
        chunks.append(new_row)

    return chunks


@st.cache_data
def get_embeddings(texts: list[str]) -> pd.DataFrame:
    """Get embeddings for a list of texts."""
    return EMBEDDING_MODEL.encode(texts)


@st.cache_data
def fit_bertopic(
    docs: list[str],
    embeddings: np.ndarray,
    zeroshot_topic_list: list[str] | None = None,
) -> tuple[BERTopic, list[int]]:
    # Initialize topic model
    topic_model = BERTopicModel()

    # Train topic model
    topic_model_output = topic_model.fit(
        docs=docs,
        embedding_model=EMBEDDING_MODEL,
        embeddings=embeddings,
        zeroshot_topic_list=zeroshot_topic_list,
        zeroshot_min_similarity=0.65,
    )

    return topic_model_output.topic_model, topic_model_output.topics


@st.cache_data
def get_improved_topic_description(
    df: pd.DataFrame, _topics_info: pd.DataFrame
) -> list[str]:
    """Get improved topic description using LLM."""
    # Get llm client
    llm_client = OpenAI_Client()

    # List of improved topics description
    improved_descriptions = []

    # Loop over topics
    for topic_number in range(len(_topics_info)):
        topic_df = df[df["topics"] == topic_number]
        user_prompt = "\n\n".join(
            topic_df.apply(
                lambda row: f"Titre : {row[TITLE_COLUMN]}\nArticle : {row[TEXT_COLUMN][0:2000]}...",
                axis=1,
            )
        )
        response = llm_client.generate(
            user_prompt=user_prompt,
            system_prompt=TOPIC_DESCRIPTION_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
        )
        improved_descriptions.append(json.loads(response)["titre"])

    return improved_descriptions


@st.cache_data
def parse_data_from_files(files: list[UploadedFile]) -> pd.DataFrame:
    """Read a list of Excel files and return a single dataframe containing the data"""
    dataframes = []

    with TemporaryDirectory() as tmpdir:
        for f in files:
            with open(tmpdir + "/" + f.name, "wb") as tmp_file:
                tmp_file.write(f.getvalue())
                print(tmp_file.name)

            if tmp_file is not None:
                with st.spinner(f"Analyse des articles de: {f.name}"):
                    provider = CurebotProvider(curebot_export_file=Path(tmp_file.name))
                    articles = provider.get_articles()
                    articles_path = Path(tmpdir) / (f.name + ".jsonl")
                    provider.store_articles(articles, articles_path)
                    df = (
                        load_data(articles_path)
                        .sort_values(by=TIMESTAMP_COLUMN, ascending=False)
                        .reset_index(drop=True)
                        .reset_index()
                    )
                    dataframes.append(df)

        # Concat all dataframes
        df_concat = pd.concat(dataframes, ignore_index=True)
        df_concat = df_concat.drop_duplicates(subset=URL_COLUMN, keep="first")
        return df_concat


@st.cache_data
def parse_data_from_feed(feed_url):
    """Return a single dataframe containing the data obtained from the feed"""
    with TemporaryDirectory() as tmpdir:
        with st.spinner(f"Analyse des articles de: {feed_url}"):
            provider = CurebotProvider(feed_url=feed_url)
            articles = provider.get_articles()
            articles_path = Path(tmpdir) / "feed.jsonl"
            provider.store_articles(articles, articles_path)
            return (
                load_data(articles_path)
                .sort_values(by=TIMESTAMP_COLUMN, ascending=False)
                .reset_index(drop=True)
                .reset_index()
            )


@st.cache_data
def update_topics_per_document(
    df: pd.DataFrame, df_split: pd.DataFrame, topics: list[int]
):
    """
    Function called after BERTopic is trained on split data.
    Find most frequent topic per document and return the updated dataframe.
    Args:
        df (pd.DataFrame): Original dataframe.
        df_split (pd.DataFrame): Split dataframe.
        topics (list[int]): List of topics for each split document.
    Returns:
        pd.DataFrame: Updated dataframe with topics per document.
    """
    # Update split dataframe with topics
    df_split["topics"] = topics

    # Group by URL (serves as a document ID) and get the most frequent topic per document
    df_topic = (
        df_split.groupby(URL_COLUMN)["topics"]
        .agg(lambda x: x.value_counts().idxmax())  # Get the most frequent topic
        .reset_index()
    )

    # Update original dataframe with topics
    df = df.merge(df_topic, on=URL_COLUMN, how="left")

    return df


def display_source_distribution(
    representative_df: pd.DataFrame, selected_sources: list[str]
):
    """Display the distribution of sources in a pie chart."""

    source_counts = representative_df[URL_COLUMN].apply(get_website_name).value_counts()

    # Create a list to store the 'pull' values for each slice
    pull = []

    # Determine which slices should be pulled out
    for source in source_counts.index:
        if source in selected_sources and "All" not in selected_sources:
            pull.append(0.2)
        else:
            pull.append(0)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=source_counts.index,
                values=source_counts.values,
                pull=pull,
                textposition="inside",
                textinfo="percent+label",
                hole=0.3,
            )
        ]
    )

    fig.update_layout(
        showlegend=False, height=600, width=500, margin=dict(t=0, b=0, l=0, r=0)
    )

    st.plotly_chart(fig, config=PLOTLY_BUTTON_SAVE_CONFIG, use_container_width=True)


def get_website_name(url):
    """Extract website name from URL, handling None, NaN, and invalid URLs."""
    if pd.isna(url) or url is None or isinstance(url, float):
        return "Unknown Source"
    try:
        return (
            urlparse(str(url)).netloc.replace("www.", "").split(".")[0]
            or "Unknown Source"
        )
    except:
        return "Unknown Source"


def display_representative_documents(filtered_df: pd.DataFrame):
    """Display representative documents for the selected topic."""
    with st.container(border=False, height=600):
        for _, doc in filtered_df.iterrows():
            website_name = get_website_name(doc.url)
            date = doc.timestamp.strftime("%A %d %b %Y %H:%M:%S")
            snippet = doc.text[:200] + "..." if len(doc.text) > 150 else doc.text

            content = f"""**{doc[TITLE_COLUMN]}**\n\n{date} | {'Unknown Source' if website_name == 'Unknown Source' else website_name}\n\n{snippet}"""

            if website_name != "Unknown Source":
                st.link_button(content, doc.url)
            else:
                st.markdown(content)
