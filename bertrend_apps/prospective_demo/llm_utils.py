#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import json

from loguru import logger

from bertrend import LLM_CONFIG
from bertrend.llm_utils.openai_client import OpenAI_Client
from bertrend.topic_analysis.prompts import TOPIC_DESCRIPTION_PROMPT


def generate_bertrend_topic_description(
    topic_words: str,
    topic_number: int,
    texts: list[str],
    language_code: str = "fr",
) -> dict:
    """Generates a LLM-based human-readable description of a topic composed of a title and a description (as a dict)"""
    if not texts:
        logger.warning(f"No text found for topic number {topic_number}")
        return {"title": "", "description": ""}

    topic_representation = ", ".join(topic_words.split("_"))  # Get top 10 words

    # Prepare the documents text
    docs_text = "\n\n".join(
        [f"Document {i + 1}: {doc[0:2000]}..." for i, doc in enumerate(texts)]
    )

    # Prepare the prompt
    prompt = TOPIC_DESCRIPTION_PROMPT[language_code]
    try:
        client = OpenAI_Client(
            api_key=LLM_CONFIG["api_key"],
            endpoint=LLM_CONFIG["endpoint"],
            model=LLM_CONFIG["model"],
        )
        answer = client.generate(
            response_format={"type": "json_object"},
            user_prompt=prompt.format(
                topic_representation=topic_representation,
                docs_text=docs_text,
            ),
        )
        return json.loads(answer)
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return f"Error generating description: {str(e)}"
