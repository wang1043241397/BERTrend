#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import pandas as pd
from bertopic import BERTopic


def get_most_representative_docs(
    topic_model: BERTopic,
    df: pd.DataFrame,
    topics,
    mode="cluster_probability",
    df_split: pd.DataFrame = None,
    topic_number: int = 0,
    top_n_docs: int = 3,
) -> pd.DataFrame:
    """
    Return most representative documents for a given topic.

    - If df_split is not None :
        Groups split docs by title to count how many paragraphs of the initial document belong to the topic.
        Returns docs having the most occurrences.

    - If df_split is None:
        Uses mode to determine the method used. Currently, support:
            * cluster_probability : computes the probability for each doc to belong to the topic using the clustering model. Returns most likely docs.
            * ctfidf_representation : computes c-TF-IDF representation for each doc and compares it to topic c-TF-IDF vector using cosine similarity. Returns highest similarity scores docs.

    """
    # If df_split is not None :
    if isinstance(df_split, pd.DataFrame):
        # Filter docs belonging to the specific topic
        sub_df = df_split.loc[pd.Series(topics) == topic_number]
        # The most representative docs in a topic are those with the highest number of extracts in this topic
        sub_df = (
            sub_df.groupby(["title"])
            .size()
            .reset_index(name="counts")
            .sort_values("counts", ascending=False)
            .iloc[0:top_n_docs]
        )
        return df[df["title"].isin(sub_df["title"])]

    # If no df_split is None, use mode to determine how to return the most representative docs:
    elif mode == "cluster_probability":
        docs_prob = topic_model.get_document_info(df["text"])["Probability"]
        df = df.assign(Probability=docs_prob)
        sub_df = df.loc[pd.Series(topics) == topic_number]
        sub_df = sub_df.sort_values("Probability", ascending=False).iloc[0:top_n_docs]
        return sub_df

    elif mode == "ctfidf_representation":
        # TODO : "get_representative_docs" currently returns maximum 3 docs as implemented in BERTopic. We should modify the function to return more if needed
        docs = topic_model.get_representative_docs(topic=topic_number)
        sub_df = df[df["text"].isin(docs)].iloc[0:top_n_docs]
        return sub_df
    return None
