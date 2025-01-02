#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
from typing import List, Tuple, Union

import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from loguru import logger
from nltk.corpus import stopwords
from numpy import ndarray
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from bertrend import BASE_CACHE_PATH, LLM_CONFIG
from bertrend.parameters import STOPWORDS
from bertrend.llm_utils.openai_client import OpenAI_Client
from bertrend.services.embedding_service import (
    convert_to_numpy,
    group_tokens,
    EmbeddingService,
)
from bertrend.utils.data_loading import TEXT_COLUMN
from bertrend.llm_utils.prompts import BERTOPIC_FRENCH_TOPIC_REPRESENTATION_PROMPT

from bertrend.utils.cache_utils import load_embeddings, save_embeddings, get_hash

# Parameters:
DEFAULT_NR_TOPICS = 10


DEFAULT_CTFIDF_MODEL = ClassTfidfTransformer(reduce_frequent_words=True)

RepresentationModelType = Union[KeyBERTInspired, MaximalMarginalRelevance, OpenAI]
DEFAULT_REPRESENTATION_MODEL: List[RepresentationModelType] = [
    MaximalMarginalRelevance(diversity=0.3)
]

# TODO - a lot of duplicate code with weak_signals - to be unified


def train_BERTopic(
    full_dataset: pd.DataFrame,
    indices: pd.Series = None,
    column: str = TEXT_COLUMN,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    umap_model: UMAP = DEFAULT_UMAP_MODEL,
    hdbscan_model: HDBSCAN = DEFAULT_HDBSCAN_MODEL,
    vectorizer_model: CountVectorizer = DEFAULT_VECTORIZER_MODEL,
    ctfidf_model: ClassTfidfTransformer = DEFAULT_CTFIDF_MODEL,
    representation_model: List[RepresentationModelType] = DEFAULT_REPRESENTATION_MODEL,
    top_n_words: int = STOPWORDS,
    nr_topics: Union[str, int] = DEFAULT_NR_TOPICS,
    use_cache: bool = True,
    cache_base_name: str = None,
    form_parameters: dict = None,
    **kwargs,
) -> Tuple[BERTopic, List[int], ndarray, ndarray, List[ndarray], List[List[str]]]:
    """
    Train a BERTopic model with customizable representation models.

    Parameters:
    ----------
    full_dataset: pd.DataFrame
        The full dataset to train
    indices: pd.Series
        Indices of the full_dataset to be used for partial training
    column: str
        Column name containing the text data
    embedding_model_name: str
        Name of the embedding model to use
    umap_model: UMAP
        UMAP model to be used in BERTopic
    hdbscan_model: HDBSCAN
        HDBSCAN model to be used in BERTopic
    vectorizer_model: CountVectorizer
        CountVectorizer model to be used in BERTopic
    ctfidf_model: ClassTfidfTransformer
        ClassTfidfTransformer model to be used in BERTopic
    representation_model: List[RepresentationModelType]
        List of representation models to use
    top_n_words: int
        Number of descriptive words per topic
    nr_topics: int
        Number of topics
    use_cache: bool
        Parameter to decide to store embeddings of the full dataset in cache
    cache_base_name: str
        Base name of the cache
    form_parameters: dict
        Additional parameters passed from the Streamlit form

    Returns:
    -------
    A tuple containing:
        - a topic model
        - a list of topics indices corresponding to the documents
        - an array of probabilities
        - the document embeddings
        - the token embeddings of each document
        - the tokens (str) of each documents
    """
    if form_parameters:
        # Update parameters based on form_parameters
        embedding_model_name = form_parameters["embedding_model_name"]
        umap_model = UMAP(
            n_neighbors=form_parameters["umap_n_neighbors"],
            n_components=form_parameters["umap_n_components"],
            min_dist=form_parameters["umap_min_dist"],
            metric=form_parameters["umap_metric"],
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=form_parameters["hdbscan_min_cluster_size"],
            min_samples=form_parameters["hdbscan_min_samples"],
            metric=form_parameters["hdbscan_metric"],
            cluster_selection_method=form_parameters[
                "hdbscan_cluster_selection_method"
            ],
            cluster_selection_epsilon=form_parameters[
                "hdbscan_cluster_selection_epsilon"
            ],
            max_cluster_size=form_parameters["hdbscan_max_cluster_size"],
            allow_single_cluster=form_parameters["hdbscan_allow_single_cluster"],
            prediction_data=True,
        )
        stop_words = (
            stopwords.words("english")
            if form_parameters["countvectorizer_stop_words"] == "english"
            else STOPWORDS
        )
        vectorizer_model = CountVectorizer(
            stop_words=stop_words,
            ngram_range=form_parameters["countvectorizer_ngram_range"],
            min_df=form_parameters["countvectorizer_min_df"],
        )
        ctfidf_model = ClassTfidfTransformer(
            reduce_frequent_words=form_parameters["ctfidf_reduce_frequent_words"],
            bm25_weighting=form_parameters["ctfidf_bm25_weighting"],
        )
        representation_model = []
        for model in form_parameters["representation_model"]:
            if model == "MaximalMarginalRelevance":
                representation_model.append(
                    MaximalMarginalRelevance(
                        diversity=form_parameters["MaximalMarginalRelevance_diversity"],
                        top_n_words=form_parameters[
                            "MaximalMarginalRelevance_top_n_words"
                        ],
                    )
                )
            elif model == "KeyBERTInspired":
                representation_model.append(
                    KeyBERTInspired(
                        top_n_words=form_parameters["KeyBERTInspired_top_n_words"],
                        nr_repr_docs=form_parameters["KeyBERTInspired_nr_repr_docs"],
                        nr_candidate_words=form_parameters[
                            "KeyBERTInspired_nr_candidate_words"
                        ],
                    )
                )
            elif model == "OpenAI":
                representation_model.append(
                    OpenAI(
                        client=OpenAI_Client(
                            api_key=LLM_CONFIG["api_key"],
                            endpoint=LLM_CONFIG["endpoint"],
                            model=LLM_CONFIG["model"],
                        ).llm_client,
                        model=os.environ["OPENAI_DEFAULT_MODEL_NAME"],
                        nr_docs=form_parameters["OpenAI_nr_docs"],
                        prompt=(
                            BERTOPIC_FRENCH_TOPIC_REPRESENTATION_PROMPT
                            if form_parameters.get("OpenAI_language", "Français")
                            == "Français"
                            else None
                        ),
                        chat=True,
                    )
                )
        top_n_words = form_parameters["bertopic_top_n_words"]
        nr_topics = (
            form_parameters["bertopic_nr_topics"]
            if form_parameters["bertopic_nr_topics"] > 0
            else None
        )
        use_cache = form_parameters["use_cached_embeddings"]

    if use_cache and cache_base_name is None:
        cache_base_name = get_hash(full_dataset[column])

    cache_path = BASE_CACHE_PATH / f"{embedding_model_name}_{cache_base_name}.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Using cache: {use_cache}")
    embedding_model = EmbeddingService(embedding_model_name)

    if indices is not None:
        filtered_dataset = full_dataset[full_dataset.index.isin(indices)].reset_index(
            drop=True
        )
    else:
        filtered_dataset = full_dataset

    if cache_path.exists() and use_cache:
        embeddings = load_embeddings(cache_path)
        logger.info(f"Embeddings loaded from cache file: {cache_path}")
        token_embeddings = None
        token_strings = None
    else:
        logger.info("Computing embeddings")
        output = embedding_model.embed(filtered_dataset[column])

        embeddings = [item["sentence_embedding"].detach().cpu() for item in output]
        embeddings = torch.stack(embeddings)
        embeddings = embeddings.numpy()

        token_embeddings = [item["token_embeddings"].detach().cpu() for item in output]
        token_ids = [item["input_ids"].detach().cpu() for item in output]

        token_embeddings = convert_to_numpy(token_embeddings)
        token_ids = convert_to_numpy(token_ids, type="token_id")

        tokenizer = embedding_model.embedding_model._first_module().tokenizer

        token_strings, token_embeddings = group_tokens(
            tokenizer, token_ids, token_embeddings, language="french"
        )

        if use_cache:
            save_embeddings(embeddings, cache_path)
            logger.info(f"Embeddings stored to cache file: {cache_path}")

    if nr_topics == 0:
        nr_topics = None

    # Separate OpenAI model if present
    openai_model = None
    other_models = []
    for model in representation_model:
        if isinstance(model, OpenAI):
            openai_model = model
        else:
            other_models.append(model)

    logger.debug(f"Representation models used: {other_models}")
    logger.debug(f"Using OpenAI to finetune representations: {(openai_model != None)}")

    # Build BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model.embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=other_models,
        top_n_words=top_n_words,
        nr_topics=nr_topics,
        calculate_probabilities=True,
        verbose=True,
    )

    logger.info("Fitting BERTopic...")

    topics, probs = topic_model.fit_transform(filtered_dataset[column], embeddings)

    logger.info("Reducing outliers via embeddings strategy...")
    if not topic_model._outliers:
        logger.warning("No outliers to reduce.")
        new_topics = topics
    else:
        new_topics = topic_model.reduce_outliers(
            documents=filtered_dataset[column],
            topics=topics,
            embeddings=embeddings,
            strategy="embeddings",
        )

        # BUG: here with bertopic > 0.16.2
        topic_model.update_topics(
            filtered_dataset[column],
            topics=new_topics,
            vectorizer_model=vectorizer_model,
            representation_model=other_models,
        )

    # If OpenAI model is present, apply it after reducing outliers
    if openai_model:
        logger.info("Applying OpenAI representation model...")
        backup_representation_model = topic_model.representation_model
        topic_model.update_topics(
            filtered_dataset[column],
            topics=new_topics,
            representation_model=openai_model,
        )
        topic_model.representation_model = backup_representation_model

    return topic_model, new_topics, probs, embeddings, token_embeddings, token_strings
