#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from collections import defaultdict
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
from bertopic import BERTopic
from loguru import logger
from sentence_transformers import SentenceTransformer

from bertrend.topic_model import TopicModel
from bertrend.utils import TEXT_COLUMN
from bertrend.parameters import (
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_GRANULARITY,
)
from bertrend.weak_signals.topic_modeling import preprocess_model, merge_models
from bertrend.weak_signals.weak_signals import (
    _initialize_new_topic,
    update_existing_topic,
    _apply_decay_to_inactive_topics,
)


class BERTrend:
    """
    A comprehensive trend analysis and weak signal detection tool using BERTopic.

    Key Parameters:
    - embedding_model_name: Name of the embedding model to use
    - granularity: Number of days to group documents
    - min_chars: Minimum character length for documents
    - split_by_paragraph: Whether to split documents by paragraph
    - sample_size: Number of documents to sample
    """

    def __init__(
        self,
        topic_model: TopicModel = None,
        zeroshot_topic_list: List[str] = None,
        zeroshot_min_similarity: float = 0,
    ):
        self.topic_model_parameters = (
            TopicModel() if topic_model is None else topic_model
        )
        self.zeroshot_topic_list = zeroshot_topic_list
        self.zeroshot_min_similarity = zeroshot_min_similarity

        # Variables related to time-based topic models
        self.emb_groups = None
        self.doc_groups = None
        self.topic_models = None

        # State variables
        self.merged_topics = None
        self._is_fitted = False

    def _train_by_period(
        self,
        period: int,
        group: Dict[pd.Timestamp, pd.DataFrame],
        embedding_model: SentenceTransformer,
        embeddings: np.ndarray,
    ) -> Tuple[
        BERTopic,
        List[str],
        np.ndarray,
    ]:
        """
        Train BERTopic models for a given time period from the grouped data.

        Args:
            period: int (int): indice of the time period
            grouped_data (Dict[pd.Timestamp, pd.DataFrame]): Dictionary of grouped data by timestamp.
            embedding_model (SentenceTransformer): Sentence transformer model for embeddings.
            embeddings (np.ndarray): Precomputed document embeddings.


        Returns:
            Tuple[BERTopic, List[str], np.ndarray]]:
                - topic_model: trained BERTopic models for this period.
                - doc_group: document groups for this period.
                - emb_group: document embeddings for this period.
        """
        docs = group[TEXT_COLUMN].tolist()
        embeddings_subset = embeddings[group.index]

        logger.debug(f"Processing period: {period}")
        logger.debug(f"Number of documents: {len(docs)}")

        logger.debug("Creating topic model...")
        topic_model = self.topic_model_parameters.create_topic_model(
            docs=docs,
            embedding_model=embedding_model,
            embeddings=embeddings_subset,
            zeroshot_topic_list=self.zeroshot_topic_list,
            zeroshot_min_similarity=self.zeroshot_min_similarity,
        )

        logger.debug("Topic model created successfully")

        doc_info_df = topic_model.get_document_info(docs=docs)
        doc_info_df = doc_info_df.rename(columns={"Document": "Paragraph"})
        doc_info_df = doc_info_df.merge(
            group[[TEXT_COLUMN, "document_id", "source", "url"]],
            left_on="Paragraph",
            right_on=TEXT_COLUMN,
            how="left",
        )
        doc_info_df = doc_info_df.drop(columns=[TEXT_COLUMN])

        topic_info_df = topic_model.get_topic_info()
        topic_doc_count_df = (
            doc_info_df.groupby("Topic")["document_id"]
            .nunique()
            .reset_index(name="Document_Count")
        )
        topic_sources_df = (
            doc_info_df.groupby("Topic")["source"]
            .apply(list)
            .reset_index(name="Sources")
        )
        topic_urls_df = (
            doc_info_df.groupby("Topic")["url"].apply(list).reset_index(name="URLs")
        )

        topic_info_df = topic_info_df.merge(topic_doc_count_df, on="Topic", how="left")
        topic_info_df = topic_info_df.merge(topic_sources_df, on="Topic", how="left")
        topic_info_df = topic_info_df.merge(topic_urls_df, on="Topic", how="left")

        topic_info_df = topic_info_df[
            [
                "Topic",
                "Count",
                "Document_Count",
                "Representation",
                "Name",
                "Representative_Docs",
                "Sources",
                "URLs",
            ]
        ]

        topic_model.doc_info_df = doc_info_df
        topic_model.topic_info_df = topic_info_df
        return topic_model, docs, embeddings_subset

    def train_topic_models(
        self,
        grouped_data: Dict[pd.Timestamp, pd.DataFrame],
        embedding_model: SentenceTransformer,
        embeddings: np.ndarray,
    ) -> Tuple[
        Dict[pd.Timestamp, BERTopic],
        Dict[pd.Timestamp, List[str]],
        Dict[pd.Timestamp, np.ndarray],
    ]:
        """
        Train BERTopic models for each timestamp in the grouped data.

        Args:
            grouped_data (Dict[pd.Timestamp, pd.DataFrame]): Dictionary of grouped data by timestamp.
            embedding_model (SentenceTransformer): Sentence transformer model for embeddings.
            embeddings (np.ndarray): Precomputed document embeddings.


        Returns:
            Tuple[Dict[pd.Timestamp, BERTopic], Dict[pd.Timestamp, List[str]], Dict[pd.Timestamp, np.ndarray]]:
                - topic_models: Dictionary of trained BERTopic models for each timestamp.
                - doc_groups: Dictionary of document groups for each timestamp.
                - emb_groups: Dictionary of document embeddings for each timestamp.
        """
        # TODO from topic_modelling = train_topic_models (modulo data transformation)
        # TODO rename to fit?
        topic_models = {}
        doc_groups = {}
        emb_groups = {}

        non_empty_groups = [
            (period, group) for period, group in grouped_data.items() if not group.empty
        ]

        # Set up progress bar
        # progress_bar = st.progress(0)
        # progress_text = st.empty()

        logger.debug(
            f"Starting to train topic models with zeroshot_topic_list: {self.zeroshot_topic_list}"
        )

        for i, (period, group) in enumerate(non_empty_groups):
            try:
                topic_models[period], doc_groups[period], emb_groups[period] = (
                    self._train_by_period(period, group, embedding_model, embeddings)
                )  # TODO: parallelize?
                logger.debug(f"Successfully processed period: {period}")

            except Exception as e:
                logger.error(f"Error processing period {period}: {str(e)}")
                logger.exception("Traceback:")
                continue

            # Update progress bar
            """
            progress = (i + 1) / len(non_empty_groups)
            progress_bar.progress(progress)
            progress_text.text(
                f"Training BERTopic model for {period} ({i + 1}/{len(non_empty_groups)})"
            )
            """

        logger.debug("Finished training all topic models")
        self._is_fitted = True

        self.topic_models = topic_models
        self.doc_groups = doc_groups
        self.emb_groups = emb_groups

        return topic_models, doc_groups, emb_groups

    def merge_models(
        self,
        min_similarity: int = DEFAULT_MIN_SIMILARITY,
        granularity: int = DEFAULT_GRANULARITY,
    ):
        if not self._is_fitted:
            raise RuntimeError("You must fit the BERTrend model before merging models.")

        topic_dfs = {
            period: preprocess_model(
                model, self.doc_groups[period], self.emb_groups[period]
            )
            for period, model in self.topic_models.items()
        }

        timestamps = sorted(topic_dfs.keys())
        merged_df_without_outliers = None
        all_merge_histories = []
        all_new_topics = []

        # progress_bar = st.progress(0)
        merge_df_size_over_time = []
        # SessionStateManager.set("merge_df_size_over_time", [])

        for i, (current_timestamp, next_timestamp) in enumerate(
            zip(timestamps[:-1], timestamps[1:])
        ):
            df1 = topic_dfs[current_timestamp][
                topic_dfs[current_timestamp]["Topic"] != -1
            ]
            df2 = topic_dfs[next_timestamp][topic_dfs[next_timestamp]["Topic"] != -1]

            if merged_df_without_outliers is None:
                if not (df1.empty or df2.empty):
                    (
                        merged_df_without_outliers,
                        merge_history,
                        new_topics,
                    ) = merge_models(
                        df1,
                        df2,
                        min_similarity=min_similarity,  # SessionStateManager.get("min_similarity"),
                        timestamp=current_timestamp,
                    )
            elif not df2.empty:
                (
                    merged_df_without_outliers,
                    merge_history,
                    new_topics,
                ) = merge_models(
                    merged_df_without_outliers,
                    df2,
                    min_similarity=min_similarity,  # SessionStateManager.get("min_similarity"),
                    timestamp=current_timestamp,
                )
            else:
                continue

            all_merge_histories.append(merge_history)
            all_new_topics.append(new_topics)
            merge_df_size_over_time = merge_df_size_over_time  # SessionStateManager.get("merge_df_size_over_time")
            merge_df_size_over_time.append(
                (
                    current_timestamp,
                    merged_df_without_outliers["Topic"].max() + 1,
                )
            )
            #
            # SessionStateManager.update(
            #     "merge_df_size_over_time", merge_df_size_over_time
            # )
            #
            # progress_bar.progress((i + 1) / len(timestamps))

        all_merge_histories_df = pd.concat(all_merge_histories, ignore_index=True)
        all_new_topics_df = pd.concat(all_new_topics, ignore_index=True)

        # SessionStateManager.set_multiple(
        #     merged_df=merged_df_without_outliers,
        #     all_merge_histories_df=all_merge_histories_df,
        #     all_new_topics_df=all_new_topics_df,
        # )

        # SessionStateManager.set("models_merged", True)
        self.model_merged = True

        return merged_df_without_outliers, all_merge_histories_df, all_new_topics_df

        (
            topic_sizes,
            topic_last_popularity,
            topic_last_update,
        ) = calculate_signal_popularity(all_merge_histories_df, granularity)
        # SessionStateManager.set_multiple(
        #     topic_sizes=topic_sizes,
        #     topic_last_popularity=topic_last_popularity,
        #     topic_last_update=topic_last_update,
        # )

        # TODO: update / set local vars

    # TODO: avoid parameter passing, use internal vars instead
    def calculate_signal_popularity(
        self,
        all_merge_histories_df: pd.DataFrame,
        granularity: int,
        decay_factor: float = 0.01,
        decay_power: float = 2,
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, float], Dict[int, pd.Timestamp]]:
        """
        Calculate the popularity of signals (topics) over time, accounting for merges and applying decay.

        Args:
            all_merge_histories_df (pd.DataFrame): DataFrame containing all merge histories.
            granularity (int): Granularity of the timestamps in days.
            decay_factor (float): Factor for exponential decay calculation.
            decay_power (float): Power for exponential decay calculation.

        Returns:
            Tuple[Dict[int, Dict[str, Any]], Dict[int, float], Dict[int, pd.Timestamp]]:
                - topic_sizes: Dictionary storing topic sizes and related information over time.
                - topic_last_popularity: Dictionary storing the last known popularity of each topic.
                - topic_last_update: Dictionary storing the last update timestamp of each topic.
        """
        topic_sizes = defaultdict(lambda: defaultdict(list))
        topic_last_popularity = {}
        topic_last_update = {}

        min_timestamp = all_merge_histories_df["Timestamp"].min()
        max_timestamp = all_merge_histories_df["Timestamp"].max()
        granularity_timedelta = pd.Timedelta(days=granularity)
        time_range = pd.date_range(
            start=min_timestamp.to_pydatetime(),
            end=(max_timestamp + granularity_timedelta).to_pydatetime(),
            freq=granularity_timedelta,
        )

        for current_timestamp in time_range:
            current_df = all_merge_histories_df[
                all_merge_histories_df["Timestamp"] == current_timestamp
            ]
            updated_topics = set()

            # Process active topics (those appearing in the current timestamp)
            for _, row in current_df.iterrows():
                current_topic = row["Topic1"]
                updated_topics.add(current_topic)

                if current_topic not in topic_sizes:
                    # Initialize new topic
                    _initialize_new_topic(
                        topic_sizes,
                        topic_last_popularity,
                        topic_last_update,
                        current_topic,
                        current_timestamp,
                        row,
                    )

                # Update existing topic
                update_existing_topic(
                    topic_sizes,
                    topic_last_popularity,
                    topic_last_update,
                    current_topic,
                    current_timestamp,
                    granularity_timedelta,
                    row,
                )

                # Mark the topic as updated for the next timestamp
                updated_topics.add(current_topic)

            # Apply decay to topics that weren't updated in this timestamp or the next
            next_timestamp = current_timestamp + granularity_timedelta
            next_df = all_merge_histories_df[
                all_merge_histories_df["Timestamp"] == next_timestamp
            ]
            topics_updated_next = set(next_df["Topic1"])

            _apply_decay_to_inactive_topics(
                topic_sizes,
                topic_last_popularity,
                topic_last_update,
                updated_topics,
                topics_updated_next,
                current_timestamp,
                granularity_timedelta,
                decay_factor,
                decay_power,
            )

        return topic_sizes, topic_last_popularity, topic_last_update

    #####################################################################################################
    # FIXME: WIP
    # def merge_models2(self):
    #     if not self._is_fitted:
    #         raise RuntimeError("You must fit the BERTrend model before merging models.")
    #
    #     merged_data = self._initialize_merge_data()
    #
    #     logger.info("Merging models...")
    #     for timestamp, model in self.topic_models.items():
    #         if not merged_data:
    #             merged_data = self._process_first_model(model)
    #         else:
    #             merged_data = self._merge_with_existing_data(
    #                 merged_data, model, timestamp
    #             )
    #
    #     self.merged_topics = merged_data
    #
    # def _merge_with_existing_data(
    #     self, merged_data: Dict, model: BERTopic, timestamp: pd.Timestamp
    # ) -> Dict:
    #     # Extract topics and embeddings
    #
    #     # Compute similarity between current model's topics and the merged ones
    #
    #     # Update merged_data with this model's data based on computed similarities
    #     # Implement business logic to handle merging decisions
    #     # This can involve thresholding, updating topic IDs, and merging document and metadata entries
    #
    #     # return merged_data  # Return the updated merged data
    #     pass
