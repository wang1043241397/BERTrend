#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path

from bertrend import MODELS_DIR, FEED_BASE_PATH

# Feed config path
USER_FEEDS_BASE_PATH = FEED_BASE_PATH / "users"
USER_FEEDS_BASE_PATH.mkdir(parents=True, exist_ok=True)

# Models config path
BASE_MODELS_DIR = MODELS_DIR / "prospective_demo" / "users"
INTERPRETATION_PATH = "interpretation"

# some identifiers
NOISE = "noise"
WEAK_SIGNALS = "weak_signals"
STRONG_SIGNALS = "strong_signals"
LLM_TOPIC_DESCRIPTION_COLUMN = "LLM Description"

# Models & analysis
DEFAULT_GRANULARITY = 2
DEFAULT_WINDOW_SIZE = 7

DEFAULT_ANALYSIS_CFG = {
    "model_config": {
        "granularity": DEFAULT_GRANULARITY,
        "window_size": DEFAULT_WINDOW_SIZE,
        "language": "French",
    },
    "analysis_config": {
        "topic_evolution": True,
        "evolution_scenarios": True,
        "multifactorial_analysis": True,
    },
}


def get_user_feed_path(user_name: str, feed_id: str) -> Path:
    feed_path = USER_FEEDS_BASE_PATH / user_name / f"{feed_id}_feed.toml"
    return feed_path


def get_user_models_path(user_name: str, model_id: str) -> Path:
    # Path to previously saved models for those data and this user
    models_path = BASE_MODELS_DIR / user_name / model_id
    models_path.mkdir(parents=True, exist_ok=True)
    return models_path


def get_model_cfg_path(user_name: str, model_id: str) -> Path:
    model_cfg_path = (
        get_user_models_path(user_name=user_name, model_id=model_id)
        / f"analysis_{model_id}.toml"
    )
    return model_cfg_path
