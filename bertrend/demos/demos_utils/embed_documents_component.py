#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import streamlit as st

from bertrend.demos.demos_utils.icons import SUCCESS_ICON
from bertrend.demos.demos_utils.messages import EMBEDDINGS_CALCULATED_MESSAGE
from bertrend.demos.demos_utils.session_state_manager import SessionStateManager
from bertrend.services.embedding_service import EmbeddingService
from bertrend.utils.data_loading import TEXT_COLUMN


def display_embed_documents_component():
    # Embed documents
    if st.button("Embed Documents"):
        with st.spinner("Embedding documents..."):
            embedding_dtype = SessionStateManager.get("embedding_dtype")
            embedding_model_name = SessionStateManager.get("embedding_model_name")
            if SessionStateManager.get("embedding_service_type", "local") == "local":
                embedding_service = EmbeddingService(
                    local=True,
                    model_name=embedding_model_name,
                    embedding_dtype=embedding_dtype,
                )
            else:
                embedding_service = EmbeddingService(
                    local=False,
                    host=SessionStateManager.get("embedding_service_hostname"),
                    port=SessionStateManager.get("embedding_service_port"),
                )

            texts = SessionStateManager.get_dataframe("time_filtered_df")[
                TEXT_COLUMN
            ].tolist()

            embedding_model, embeddings, token_strings, token_embeddings = (
                embedding_service.embed(
                    texts=texts,
                )
            )

            SessionStateManager.set("embedding_model", embedding_model)
            SessionStateManager.set("embeddings", embeddings)
            SessionStateManager.set("token_strings", token_strings)
            SessionStateManager.set("token_embeddings", token_embeddings)

            SessionStateManager.set("data_embedded", True)

            st.success(EMBEDDINGS_CALCULATED_MESSAGE, icon=SUCCESS_ICON)
