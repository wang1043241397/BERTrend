#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import typer
from loguru import logger

from bertrend.BERTopicModel import BERTopicModel
from bertrend.BERTrend import BERTrend
from bertrend.services.embedding_service import EmbeddingService
from bertrend.utils.data_loading import split_data, group_by_days, TEXT_COLUMN
from bertrend_apps.prospective_demo import get_user_models_path
from bertrend_apps.prospective_demo.process_new_data import (
    load_all_data,
    get_relevant_model_config,
)


# Script to regenerate past models from scratch for a given feed
# Use retrospective analysis methodology
# This will not regenerate the analysis which are quite costly


def regenerate_models(model_id: str, user: str):
    """Regenerate from scratch (method retrospective) the models associated to the specified model identifier
    for the specified user."""

    # Get relevant model info from config
    granularity, window_size, language = get_relevant_model_config(
        model_id=model_id, user=user
    )

    # Path to saved models
    bertrend_models_path = get_user_models_path(user, model_id)

    # Initialization of embedding service
    # TODO: customize service (lang, etc)
    embedding_service = EmbeddingService(local=False)

    # Load model config
    df = load_all_data(model_id=model_id, user=user, language=language)
    logger.info(f"Size of dataset: {len(df)}")

    # Split data by paragraphs
    df = split_data(df)

    # Process new data and save models
    # - Group data based on granularity
    grouped_data = group_by_days(df=df, day_granularity=granularity)

    # Train BERTrend
    bertrend = BERTrend(topic_model=BERTopicModel({"global": {"language": language}}))
    embeddings, _, _ = embedding_service.embed(
        texts=df[TEXT_COLUMN],
    )
    bertrend.train_topic_models(
        grouped_data=grouped_data,
        embedding_model=embedding_service.embedding_model_name,
        embeddings=embeddings,
        bertrend_models_path=bertrend_models_path,
        save_topic_models=True,
    )
    bertrend.save_model(models_path=bertrend_models_path)

    logger.success(
        f"Regenerated models for '{model_id}' from scratch. BERTrend model was built using {len(bertrend.doc_groups)} models/time periods."
    )


if __name__ == "__main__":
    app = typer.Typer()

    @app.command("regenerate")
    def regenerate(
        model_id: str = typer.Argument(
            help="identifier of the model to be regenerated from scratch"
        ),
        user: str = typer.Argument(default="nemo", help="identifier of the user"),
    ):
        """Regenerate past models"""
        regenerate_models(model_id=model_id, user=user)

    # Main app
    app()
