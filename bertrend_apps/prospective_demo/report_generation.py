#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import inspect
import tempfile
import re

import pandas as pd
import streamlit as st
from pathlib import Path

from google.auth.exceptions import RefreshError
from jinja2 import FileSystemLoader, Environment
from loguru import logger
from tqdm import tqdm

from bertrend.demos.demos_utils.i18n import translate
from bertrend.demos.demos_utils.icons import (
    NEWSLETTER_ICON,
    TOPIC_ICON,
    ERROR_ICON,
    DOWNLOAD_ICON,
    EMAIL_ICON,
)
from bertrend.llm_utils.newsletter_features import generate_newsletter
from bertrend.llm_utils.newsletter_model import (
    STRONG_TOPIC_TYPE,
    WEAK_TOPIC_TYPE,
    Article,
)
from bertrend.trend_analysis.data_structure import TopicSummaryList, SignalAnalysis
from bertrend_apps.common.mail_utils import get_credentials, send_email
from bertrend_apps.prospective_demo import (
    WEAK_SIGNALS,
    STRONG_SIGNALS,
    LLM_TOPIC_DESCRIPTION_COLUMN,
    LLM_TOPIC_TITLE_COLUMN,
    URLS_COLUMN,
)
from bertrend_apps.prospective_demo.dashboard_common import choose_id_and_ts
from bertrend_apps.prospective_demo.data_model import DetailedNewsletter, TopicOverTime

WEAK_SIGNAL_NB = 3
STRONG_SIGNAL_NB = 5

MAXIMUM_NUMBER_OF_ARTICLES = 3


def reporting():
    choose_id_and_ts()

    tab1, tab2 = st.tabs(
        [
            TOPIC_ICON + " " + translate("step_1_title"),
            NEWSLETTER_ICON + " " + translate("step_2_title"),
        ]
    )
    with tab1:
        selected_weak_topics_df, selected_strong_topics_df = choose_topics()
    with tab2:
        configure_export(selected_weak_topics_df, selected_strong_topics_df)


def choose_topics():
    st.subheader(translate("step_1_subheader"))
    model_id = st.session_state.model_id
    dfs_interpretation = st.session_state.signal_interpretations
    if model_id not in dfs_interpretation:
        st.error(f"{ERROR_ICON} {translate('no_data')}")
        st.stop()
    cols = st.columns(2)
    with cols[0]:
        st.write(f"#### :orange[{translate('emerging_topics')}]")
        if WEAK_SIGNALS not in dfs_interpretation[model_id]:
            st.error(f"{ERROR_ICON} {translate('no_data')}")
            filtered_weak_signals = None
        else:
            df_w = dfs_interpretation[model_id][WEAK_SIGNALS]
            weak_topics_list = choose_from_df(df_w)
            filtered_weak_signals = df_w[df_w["Topic"].isin(weak_topics_list)]

    with cols[1]:
        st.write(f"#### :green[{translate('strong_topics')}]")
        if STRONG_SIGNALS not in dfs_interpretation[model_id]:
            st.error(f"{ERROR_ICON} {translate('no_data')}")
            filtered_strong_signals = None
        else:
            df_w = dfs_interpretation[model_id][STRONG_SIGNALS]
            strong_topics_list = choose_from_df(df_w)
            filtered_strong_signals = df_w[df_w["Topic"].isin(strong_topics_list)]

    return filtered_weak_signals, filtered_strong_signals


def choose_from_df(df: pd.DataFrame):
    df["A retenir"] = True
    df["Sujet"] = df[LLM_TOPIC_TITLE_COLUMN]
    df["Description"] = df[LLM_TOPIC_DESCRIPTION_COLUMN]
    columns = ["Topic", "A retenir", "Sujet", "Description"]
    pd.DataFrame(
        [
            {"command": "st.selectbox", "rating": 4, "is_widget": True},
            {"command": "st.balloons", "rating": 5, "is_widget": False},
            {"command": "st.time_input", "rating": 3, "is_widget": True},
        ]
    )
    edited_df = st.data_editor(df[columns], num_rows="dynamic", column_order=columns)
    selection = edited_df[edited_df["A retenir"] == True]["Topic"].tolist()
    return selection


def configure_export(weak_signals: pd.DataFrame, strong_signals: pd.DataFrame):
    st.subheader(translate("step_2_subheader"))
    st.write(translate("export_configuration_note"))

    st.button(
        translate("generate_button_label"),
        type="primary",
        on_click=lambda: generate_report(weak_signals, strong_signals),
    )


def create_detailed_newsletter(
    weak_signals: pd.DataFrame, strong_signals: pd.DataFrame
) -> DetailedNewsletter:
    model_id = st.session_state.model_id

    # Create the DetailedNewsletter object
    detailed_newsletter = DetailedNewsletter(
        title=model_id,
        reference_period=st.session_state.reference_ts.date(),
        topics=[],
    )

    # Iterate over topics
    for df, topic_type in tqdm(
        zip([weak_signals, strong_signals], [STRONG_TOPIC_TYPE, WEAK_TOPIC_TYPE])
    ):
        # Iterate over the filtered DataFrame rows
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            # For each row, iterate over each column
            articles = [
                Article(title=link, date=None, source=None, url=link)
                for link in list(set(row[URLS_COLUMN]))[:MAXIMUM_NUMBER_OF_ARTICLES]
            ]

            topic: TopicOverTime = TopicOverTime(
                title=row[LLM_TOPIC_TITLE_COLUMN],
                hashtags=None,
                summary=row[LLM_TOPIC_DESCRIPTION_COLUMN],
                articles=articles,
                topic_type=topic_type,
                topic_evolution=TopicSummaryList.model_validate_json(row["summary"]),
                topic_analysis=SignalAnalysis.model_validate_json(
                    row["analysis"],
                ),
            )

            detailed_newsletter.topics.append(topic)

    return detailed_newsletter


@st.dialog(translate("report_preview_title"), width="large")
def generate_report(weak_signals: pd.DataFrame, strong_signals: pd.DataFrame):

    detailed_newsletter: DetailedNewsletter = create_detailed_newsletter(
        weak_signals, strong_signals
    )

    # Generate the HTML
    output_html = render_html_report(
        newsletter=detailed_newsletter,
        language=st.session_state.internationalization_language,
    )

    # Render HTML
    with st.container(height=475):
        st.html(output_html)

    # Save report to temp file
    temp_report_path = create_temp_report(output_html)  # Create the file in temp dir

    cols = st.columns([2, 3])
    model_id = st.session_state.model_id
    with cols[0]:
        download(temp_report_path, model_id)
    with cols[1]:
        email(
            temp_report_path, mail_title=f"{translate('report_mail_title')} {model_id}"
        )


def render_html_report(
    newsletter: DetailedNewsletter,
    language: str = "fr",
) -> str:

    template_dirs = [
        Path(__file__).parent,  # Current directory
        Path(
            inspect.getfile(generate_newsletter)
        ).parent,  # Main template ("newsletter_outlook_template.html")
    ]

    # Set up the Jinja2 environment to look in both directories
    env = Environment(loader=FileSystemLoader(template_dirs))

    # Render the template with data
    template = env.get_template("detailed_report_template.html")
    rendered_html = template.render(
        newsletter=newsletter, language=language, custom_css=""
    )

    return rendered_html


def create_temp_report(html_content) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False
    ) as temp_file:
        temp_file.write(html_content)
        return Path(temp_file.name)


def download(temp_path: Path, model_id: str):
    with open(temp_path, "r", encoding="utf-8") as file:
        st.download_button(
            label=f"{DOWNLOAD_ICON} {translate('download_button_label')}",
            type="primary",
            data=file.read(),
            file_name=f"rapport_{model_id}.html",
            mime="text/html",
        )


def is_valid_email(email: str) -> bool:
    """Checks if an email address is valid using a regular expression."""
    regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(regex, email) is not None


def email(temp_path: Path, mail_title: str) -> None:
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        user_email = st.text_input("@", label_visibility="collapsed")

    with col2:
        if st.button(f"{EMAIL_ICON} {translate('send_button_label')}", type="primary"):
            if not user_email:
                return
            if not is_valid_email(user_email):
                st.error(f"{ERROR_ICON} {translate('invalid_email')}")
                return

            try:
                # Send the newsletter by email
                # string to list conversion for recipients
                recipients = [user_email]
                try:
                    if recipients:
                        credentials = get_credentials()
                        with open(temp_path, "r") as file:
                            # Read the entire contents of the file into a string
                            content = file.read()
                        send_email(credentials, mail_title, recipients, content, "html")
                except RefreshError as re:
                    logger.error(
                        f"Problem with token for email, please regenerate it: {re}"
                    )

                st.success(translate("email_sent_successfully"))

            except Exception as e:
                st.error(f"{translate('email_error_message')}: {e}")
