#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from datetime import timedelta
from pathlib import Path

import pandas as pd
import typer
from jsonlines import jsonlines
from loguru import logger

from bertrend import load_toml_config, FEED_BASE_PATH
from bertrend.BERTrend import train_new_data, BERTrend
from bertrend.services.embedding_service import EmbeddingService
from bertrend.topic_analysis.topic_description import generate_topic_description
from bertrend.trend_analysis.weak_signals import analyze_signal
from bertrend.utils.data_loading import load_data, split_data
from bertrend_apps.prospective_demo import (
    get_user_feed_path,
    get_user_models_path,
    INTERPRETATION_PATH,
)

DEFAULT_TOP_K = 5

if __name__ == "__main__":
    app = typer.Typer()

    @app.command("train-new-model")
    def train_new_model(
        user_name: str = typer.Argument(help="Identifier of the user"),
        model_id: str = typer.Argument(help="ID of the model/data to train"),
        language: str = typer.Option(
            help="The language to use for the model ('French' or 'English')",
            default="French",
        ),
        granularity: int = typer.Option(
            help="The granularity to use for the model (in days)", default=7
        ),
        window_size: int = typer.Option(
            help="The window size for analysis (in days)", default=7
        ),
    ):
        if language not in ["French", "English"]:
            language = "French"
        language_code = "fr" if language == "French" else "en"

        # Path to previously saved models for those data and this user
        bertrend_models_path = get_user_models_path(user_name, model_id)

        # Initialization of embedding service
        # TODO: customize service (lang, etc)
        embedding_service = EmbeddingService(local=True)

        # load data for last period
        # TODO: to be improved
        cfg_file = get_user_feed_path(user_name, model_id)
        if not cfg_file.exists():
            logger.error(f"Cannot find/process config file: {cfg_file}")
            return
        cfg = load_toml_config(cfg_file)
        feed_base_dir = cfg["data-feed"]["feed_dir_path"]
        files = list(
            Path(FEED_BASE_PATH, feed_base_dir).glob(
                f"*{cfg['data-feed'].get('id')}*.jsonl*"
            )
        )

        if not files:
            logger.warning(f"No new data for '{model_id}', nothing to do")
            return

        dfs = [load_data(Path(f), language=language) for f in files]
        new_data = pd.concat(dfs).drop_duplicates(
            subset=["title"], keep="first", inplace=False
        )

        # filter data according to granularity
        # Calculate the date X days ago
        reference_timestamp = pd.Timestamp(
            new_data["timestamp"].max().date()
        )  # used to identify the last model
        cut_off_date = new_data["timestamp"].max() - timedelta(days=granularity)
        # Filter the DataFrame to keep only the rows within the last X days
        filtered_df = new_data[new_data["timestamp"] >= cut_off_date]

        filtered_df = split_data(filtered_df)

        logger.info(f'Processing new data for user "{user_name}" about "{model_id}"...')
        # Process new data
        bertrend = train_new_data(
            filtered_df,
            bertrend_models_path=bertrend_models_path,
            embedding_service=embedding_service,
        )

        if not bertrend._are_models_merged:
            # This is generally the case when we have only one model
            return

        # Compute popularities
        bertrend.calculate_signal_popularity()

        # classify last signals
        noise_topics_df, weak_signal_topics_df, strong_signal_topics_df = (
            bertrend.classify_signals(window_size, cut_off_date)
        )

        # enrich signal description with LLM-based topic description
        for df in [noise_topics_df, weak_signal_topics_df, strong_signal_topics_df]:
            df["llm_topic_description"] = df["Topic"].apply(
                lambda topic: generate_topic_description(
                    topic_model=bertrend.topic_models[reference_timestamp],
                    topic_number=topic,
                    filtered_docs=filtered_df,
                    language_code=language_code,
                )
            )

        # Store signals
        for df, df_name in zip(
            [noise_topics_df, weak_signal_topics_df, strong_signal_topics_df],
            ["noise_topics", "weak_signals", "strong_signals"],
        ):
            df.to_parquet(
                f"{INTERPRETATION_PATH}/{reference_timestamp}/{df_name}.parquet"
            )
            generate_llm_interpretation(bertrend, reference_timestamp, df, df_name)

    def generate_llm_interpretation(
        bertrend: BERTrend,
        reference_timestamp: pd.Timestamp,
        df: pd.DataFrame,
        df_name: str,
        top_k: int = DEFAULT_TOP_K,
    ):
        """Generate detailed analysis for the top k (using template)"""

        interpretation = []
        for topic in df.head(top_k)["Topic"]:
            summary, analysis, formatted_html = analyze_signal(
                bertrend, topic, reference_timestamp
            )
            interpretation.append(
                {"topic": topic, "summary": summary, "analysis": formatted_html}
            )

        # Save interpretation
        with jsonlines.open(
            f"{INTERPRETATION_PATH}/{reference_timestamp}/{df_name}_interpretation.jsonl",
            mode="w",
        ) as writer:
            for item in interpretation:
                writer.write(item)

    # Main app
    app()
