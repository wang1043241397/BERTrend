#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from typing import List
from pydantic import BaseModel
from datetime import date

STRONG_TOPIC_TYPE = "strong"
WEAK_TOPIC_TYPE = "weak"
NOISE_TOPIC_TYPE = "noise"


class Article(BaseModel):
    """Article model."""

    title: str
    date: date
    summary: str | None = None
    source: str | None = None
    url: str | None = None


class Topic(BaseModel):
    """Topic model."""

    title: str
    hashtags: List[str]
    summary: str | None = None
    articles: List[Article]
    topic_type: str = (
        STRONG_TOPIC_TYPE  # Default value "strong", can be "weak" or "noise"
    )


class Newsletter(BaseModel):
    """Newsletter model."""

    title: str
    period_start_date: date
    period_end_date: date
    topics: List[Topic]
    debug_info: dict | None = None
