#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path
from typing import Tuple, List, Literal

import numpy as np
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from bertrend import load_toml_config
from bertrend.parameters import (
    DEFAULT_UMAP_N_COMPONENTS,
    DEFAULT_UMAP_N_NEIGHBORS,
    DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    DEFAULT_HDBSCAN_MIN_SAMPLES,
    HDBSCAN_CLUSTER_SELECTION_METHODS,
    VECTORIZER_NGRAM_RANGES,
    DEFAULT_MIN_DF,
    DEFAULT_TOP_N_WORDS,
    LANGUAGES,
    DEFAULT_UMAP_MIN_DIST,
    STOPWORDS,
    DEFAULT_MMR_DIVERSITY,
    OUTLIER_REDUCTION_STRATEGY,
    ENGLISH_STOPWORDS,
)


class TopicModelOutput:
    """Wrapper to encapsulate all results related to topic model output"""

    def __init__(self, topic_model: BERTopic):
        """
        - a topic model
        - a list of topics indices corresponding to the documents
        - an array of probabilities
        - the document embeddings
        - the token embeddings of each document
        - the tokens (str) of each documents
        """
        # Topic model
        self.topic_model = topic_model
        # List of topics indices corresponding to the documents
        self.topics = None
        # Array of probabilities
        self.probs = None
        # Document embeddings
        self.embeddings = None
        # Token embeddings of each document
        self.token_embeddings = None
        # Tokens (str) of each document
        self.token_strings = None


class TopicModel:
    """
    Class that encapsulates the parameters for topic models.

    Args:
        umap_n_components (int): Number of components for UMAP.
        umap_n_neighbors (int): Number of neighbors for UMAP.
        umap_min_dist (float): Minimum distance between neighbors for UMAP.
        hdbscan_min_cluster_size (int): Minimum cluster size for HDBSCAN.
        hdbscan_min_samples (int): Minimum samples for HDBSCAN.
        hdbscan_cluster_selection_method (str): Cluster selection method for HDBSCAN.
        vectorizer_ngram_range (Tuple[int, int]): N-gram range for CountVectorizer.
        min_df (int): Minimum document frequency for CountVectorizer.
        top_n_words (int): Number of top words to include in topic representations.
        zeroshot_topic_list (List[str]): List of topics for zero-shot classification.
        zeroshot_min_similarity (float): Minimum similarity threshold for zero-shot classification.
        language (str): Used to determine the list of stopwords to use
    """

    def __init__(
        self,
        umap_n_components: int = DEFAULT_UMAP_N_COMPONENTS,
        umap_n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS,
        umap_min_dist: float = DEFAULT_UMAP_MIN_DIST,
        hdbscan_min_cluster_size: int = DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
        hdbscan_min_samples: int = DEFAULT_HDBSCAN_MIN_SAMPLES,
        hdbscan_cluster_selection_method: str = HDBSCAN_CLUSTER_SELECTION_METHODS[0],
        vectorizer_ngram_range: Tuple[int, int] = VECTORIZER_NGRAM_RANGES[0],
        min_df: int = DEFAULT_MIN_DF,
        top_n_words: int = DEFAULT_TOP_N_WORDS,
        language: str = LANGUAGES[0],
        outlier_reduction_strategy: Literal[
            "c-tf-idf", "embeddings"
        ] = OUTLIER_REDUCTION_STRATEGY,
        mmr_diversity: float = DEFAULT_MMR_DIVERSITY,
    ):
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.hdbscan_cluster_selection_method = hdbscan_cluster_selection_method
        self.vectorizer_ngram_range = vectorizer_ngram_range
        self.min_df = min_df
        self.top_n_words = top_n_words
        self.language = language
        self.outlier_reduction_strategy = outlier_reduction_strategy
        self.mmr_diversity = mmr_diversity

        # Initialize models based on those parameters
        self._initialize_models()

    def _initialize_models(self):
        self.umap_model = UMAP(
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            random_state=42,
            metric="cosine",
        )

        self.hdbscan_model = HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_method=self.hdbscan_cluster_selection_method,
            prediction_data=True,
        )

        self.stopword_set = (
            STOPWORDS if self.language == "French" else ENGLISH_STOPWORDS
        )
        self.vectorizer_model = CountVectorizer(
            stop_words=self.stopword_set,
            min_df=self.min_df,
            ngram_range=self.vectorizer_ngram_range,
        )
        self.mmr_model = MaximalMarginalRelevance(diversity=self.mmr_diversity)
        self.ctfidf_model = ClassTfidfTransformer(
            reduce_frequent_words=True, bm25_weighting=False
        )

    @classmethod
    def from_config(cls, config: Path) -> "TopicModel":
        """Creates a topic model from a toml configuration file."""
        parameters = load_toml_config(config)["bertopic_parameters"]
        return cls(
            umap_n_components=parameters.get(
                "umap_n_components", DEFAULT_UMAP_N_COMPONENTS
            ),
            umap_n_neighbors=parameters.get(
                "umap_n_neighbors", DEFAULT_UMAP_N_NEIGHBORS
            ),
            umap_min_dist=parameters.get("umap_min_dist", DEFAULT_UMAP_MIN_DIST),
            hdbscan_min_cluster_size=parameters.get(
                "hdbscan_min_cluster_size", DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE
            ),
            hdbscan_min_samples=parameters.get(
                "hdbscan_min_samples", DEFAULT_HDBSCAN_MIN_SAMPLES
            ),
            hdbscan_cluster_selection_method=parameters.get(
                "hdbscan_cluster_selection_method", HDBSCAN_CLUSTER_SELECTION_METHODS[0]
            ),
            vectorizer_ngram_range=parameters.get(
                "vectorizer_ngram_range", VECTORIZER_NGRAM_RANGES[0]
            ),
            min_df=parameters.get("min_df", DEFAULT_MIN_DF),
            top_n_words=parameters.get("top_n_words", DEFAULT_TOP_N_WORDS),
            language=parameters.get("language", LANGUAGES[0]),
            outlier_reduction_strategy=parameters.get(
                "outlier_reduction_strategy", OUTLIER_REDUCTION_STRATEGY
            ),
            mmr_diversity=parameters.get("mmr_diversity", DEFAULT_MMR_DIVERSITY),
        )

    def fit(
        self,
        docs: List[str],
        embedding_model: SentenceTransformer,
        embeddings: np.ndarray,
        zeroshot_topic_list=None,
        zeroshot_min_similarity: float = 0,
    ) -> TopicModelOutput:
        """
        Create a TopicModelOutput model.

        Args:
            docs (List[str]): List of documents.
            embedding_model (SentenceTransformer): Sentence transformer model for embeddings.
            embeddings (np.ndarray): Precomputed document embeddings.
            umap_model (UMAP): UMAP model for dimensionality reduction.
            hdbscan_model (HDBSCAN): HDBSCAN model for clustering.
            vectorizer_model (CountVectorizer): CountVectorizer model for creating the document-term matrix.
            mmr_model (MaximalMarginalRelevance): MMR model for diverse topic representation.
            top_n_words (int): Number of top words to include in topic representations.
            zeroshot_topic_list (List[str]): List of topics for zero-shot classification.
            zeroshot_min_similarity (float): Minimum similarity threshold for zero-shot classification.

        Returns:
            BERTopic: A fitted BERTopic model.
        """
        if zeroshot_topic_list is None:
            zeroshot_topic_list = list()
        if zeroshot_topic_list is None:
            zeroshot_topic_list = []
        logger.debug(
            f"\tCreating topic model with zeroshot_topic_list: {zeroshot_topic_list}"
        )
        try:
            # Handle scenario where user enters a bunch of white space characters or any scenario where we can't extract zeroshot topics
            # BERTopic needs a "None" instead of an empty list, otherwise it'll attempt zeroshot topic modeling on an empty list
            if len(zeroshot_topic_list) == 0:
                zeroshot_topic_list = None

            logger.debug("\tInitializing BERTopic model")

            topic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=self.umap_model,
                hdbscan_model=self.hdbscan_model,
                vectorizer_model=self.vectorizer_model,
                ctfidf_model=self.ctfidf_model,
                representation_model=self.mmr_model,
                zeroshot_topic_list=zeroshot_topic_list,
                zeroshot_min_similarity=zeroshot_min_similarity,
            )
            logger.success("\tBERTopic model instance created successfully")

            logger.debug("\tFitting BERTopic model")
            topics, probs = topic_model.fit_transform(docs, embeddings)

            if not topic_model._outliers:
                logger.warning("\tNo outliers to reduce.")
                new_topics = topics
            else:
                logger.debug("\tReducing outliers")
                new_topics = topic_model.reduce_outliers(
                    documents=docs,
                    topics=topics,
                    embeddings=embeddings,
                    strategy=OUTLIER_REDUCTION_STRATEGY,
                )
            topic_model.update_topics(
                docs=docs,
                topics=new_topics,
                vectorizer_model=self.vectorizer_model,
                ctfidf_model=self.ctfidf_model,
                representation_model=self.mmr_model,
            )

            logger.success("\tBERTopic model fitted successfully")
            output = TopicModelOutput(topic_model)
            output.topics = new_topics
            output.probs = probs
            return output
        except Exception as e:
            logger.error(f"\tError in create_topic_model: {str(e)}")
            logger.exception("\tTraceback:")
            raise
