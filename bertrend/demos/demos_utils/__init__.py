#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import streamlit as st


def is_admin_mode() -> bool:
    """Indicates whether the application is in an admin mode."""
    admin_param = st.query_params.get("admin")
    if not admin_param:
        return False
    return admin_param.lower() == "true"
