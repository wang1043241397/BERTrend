#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import pandas as pd
from plotly import express as px


def plot_topics_over_time(
    topics_over_time, dynamic_topics_list, topic_model, time_split=None, width=900
):
    if dynamic_topics_list != "":
        if ":" in dynamic_topics_list:
            dynamic_topics_list = [
                i
                for i in range(
                    int(dynamic_topics_list.split(":")[0]),
                    int(dynamic_topics_list.split(":")[1]),
                )
            ]
        else:
            dynamic_topics_list = [int(i) for i in dynamic_topics_list.split(",")]
        fig = topic_model.visualize_topics_over_time(
            topics_over_time,
            topics=dynamic_topics_list,
            width=width,
            title="",
        )
        if time_split:
            fig.add_vline(
                x=time_split,
                line_width=3,
                line_dash="dash",
                line_color="black",
                opacity=1,
            )
        return fig


def plot_docs_repartition_over_time(df, freq):
    """
    Plot document distribution over time
    """
    count = df.groupby(pd.Grouper(key="timestamp", freq=freq), as_index=False).size()
    count["timestamp"] = count["timestamp"].dt.strftime("%Y-%m-%d")

    fig = px.bar(count, x="timestamp", y="size")
    return fig


def plot_remaining_docs_repartition_over_time(df_base, df_remaining, freq):
    """
    Plot remaining document distribution over time
    """
    df = pd.concat([df_base, df_remaining])

    # Get split time value
    split_time = str(df_remaining["timestamp"].min())

    # Print aggregated docs
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    count = df.groupby(pd.Grouper(key="timestamp", freq=freq), as_index=False).size()
    count["timestamp"] = count["timestamp"].dt.strftime("%Y-%m-%d")
    # Split to set a different color to each DF
    count["category"] = [
        "Base" if time < split_time else "Remaining" for time in count["timestamp"]
    ]

    fig = px.bar(
        count,
        x="timestamp",
        y="size",
        color="category",
        color_discrete_map={
            "Base": "light blue",  # default plotly color to match main page graphs
            "Remaining": "orange",
        },
    )
    return fig
