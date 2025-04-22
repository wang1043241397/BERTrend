#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from pydantic import BaseModel


class TopicDescription(BaseModel):
    # Title of the topic
    title: str
    # Description of the topic
    description: str
