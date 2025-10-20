#
# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of BERTrend.
#

# Starts the Summarizer app (a comparison of various method of text summarization)
CUDA_VISIBLE_DEVICES=0 OPENAI_API_KEY=$OPENAI_API_KEY OPENAI_BASE_URL=$OPENAI_BASE_URL OPENAI_DEFAULT_MODEL=$OPENAI_DEFAULT_MODEL streamlit run --theme.primaryColor royalblue summarization/summarizer_app.py