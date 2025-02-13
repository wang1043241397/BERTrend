#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import os
import tempfile
import re

import pandas as pd
import streamlit as st
from pathlib import Path

from html2docx import html2docx

from bertrend.demos.demos_utils.icons import (
    NEWSLETTER_ICON,
    TOPIC_ICON,
    ERROR_ICON,
    DOWNLOAD_ICON,
    EMAIL_ICON,
)
from bertrend_apps.prospective_demo import (
    WEAK_SIGNALS,
    STRONG_SIGNALS,
    LLM_TOPIC_DESCRIPTION_COLUMN,
    LLM_TOPIC_TITLE_COLUMN,
    URLS_COLUMN,
)
from bertrend_apps.prospective_demo.dashboard_common import choose_id_and_ts
from bertrend_apps.prospective_demo.report_utils import generate_html_report

WEAK_SIGNAL_NB = 3
STRONG_SIGNAL_NB = 5


def reporting():
    choose_id_and_ts()

    tab1, tab2 = st.tabs(
        [
            TOPIC_ICON + " Etape 1: Sélection des sujets à retenir",
            NEWSLETTER_ICON + " Etape 2: Export",
        ]
    )
    with tab1:
        selected_weak_topics_df, selected_strong_topics_df = choose_topics()
    with tab2:
        configure_export(selected_weak_topics_df, selected_strong_topics_df)

    # generate_newsletter()


def choose_topics():
    st.subheader("Etape 1: Sélection des sujets à retenir")
    model_id = st.session_state.model_id
    dfs_interpretation = st.session_state.signal_interpretations
    if model_id not in dfs_interpretation:
        st.error(f"{ERROR_ICON} Pas de données")
        st.stop()
    cols = st.columns(2)
    with cols[0]:
        st.write("#### :orange[Sujets émergents]")
        if WEAK_SIGNALS not in dfs_interpretation[model_id]:
            st.error(f"{ERROR_ICON} Pas de données")
            filtered_weak_signals = None
        else:
            df_w = dfs_interpretation[model_id][WEAK_SIGNALS]
            weak_topics_list = choose_from_df(df_w)
            filtered_weak_signals = df_w[df_w["Topic"].isin(weak_topics_list)]

    with cols[1]:
        st.write("#### :green[Sujets forts]")
        if STRONG_SIGNALS not in dfs_interpretation[model_id]:
            st.error(f"{ERROR_ICON} Pas de données")
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
    st.subheader("Etape 2: Configuration de l'export")
    st.write(
        "TODO: séléction des parties à inclure dans l'export par topic: résumé global, liens, évolution par période, analyse détaillée, etc."
    )

    st.button(
        "Générer",
        type="primary",
        on_click=lambda: generate_report(weak_signals, strong_signals),
    )


@st.dialog("Rapport (aperçu)", width="large")
def generate_report(weak_signals: pd.DataFrame, strong_signals: pd.DataFrame):
    model_id = st.session_state.model_id

    # TODO: output à configurer selon les infos choisies
    # rendering parameters
    number_links = 2

    report_template = Path(__file__).parent / "report_template.html"
    report_title = f"Actu <font color='royal_blue'>{model_id}</font>"
    data_list = []
    for df, color in zip([weak_signals, strong_signals], ["orange", "green"]):
        # Iterate over the filtered DataFrame rows
        for _, row in df.iterrows():
            # For each row, iterate over each column
            data_list.append(
                {
                    "topic_title": row[LLM_TOPIC_TITLE_COLUMN],
                    "topic_description": row[LLM_TOPIC_DESCRIPTION_COLUMN],
                    "color": color,
                    "links": list(set(row[URLS_COLUMN]))[:number_links],
                    "analysis": row["analysis"],
                }
            )

    # Generate the HTML
    output_html = generate_html_report(report_template, report_title, data_list)

    # Render HTML
    with st.container(height=475):
        st.html(output_html)

    # Save report to temp file
    temp_report_path = create_temp_report(output_html)  # Create the file in temp dir

    cols = st.columns([1, 1, 3])
    with cols[0]:
        download(temp_report_path, model_id)
    with cols[1]:
        # download_docx(temp_report_path, model_id)
        pass
    with cols[2]:
        email(temp_report_path)  #


def create_temp_report(html_content) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False
    ) as temp_file:
        temp_file.write(html_content)
        return Path(temp_file.name)


def download(temp_path: Path, model_id: str):
    with open(temp_path, "r", encoding="utf-8") as file:
        st.download_button(
            label=f"{DOWNLOAD_ICON} Télécharger (html)",
            type="primary",
            data=file.read(),
            file_name=f"rapport_{model_id}.html",
            mime="text/html",
        )


def download_docx(temp_path: Path, model_id: str):
    docx = convert_html_to_docx(temp_path)
    with open(docx, "r", encoding="utf-8") as file:
        st.download_button(
            label=f"{DOWNLOAD_ICON} Télécharger (docx)",
            type="primary",
            data=file.read(),
            file_name=f"rapport_{model_id}.html",
            mime="text/html",
        )
    os.remove(docx)


def is_valid_email(email: str) -> bool:
    """Checks if an email address is valid using a regular expression."""
    regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(regex, email) is not None


def email(temp_path: Path) -> None:
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        user_email = st.text_input("@", label_visibility="hidden")

    with col2:
        if st.button(f"{EMAIL_ICON} Envoyer", type="primary"):
            if not user_email:
                return

            if not is_valid_email(user_email):
                st.error(f"{ERROR_ICON} Adresse mail incorrecte")
                return

            try:
                # TODO

                # Implement your email sending logic here.  Use the temp_path and user_email.
                # Example (replace with your actual email code, handling attachments):
                # import smtplib

                #     server.sendmail(sender_email, user_email, message.as_string())

                st.success("Email sent successfully!")

            except Exception as e:
                st.error(f"Error sending email: {e}")


def convert_html_to_docx(html_path: Path, output_path=None):
    """
    Convert HTML file to DOCX format.

    Args:
        html_path (Path): Path to the input HTML file
        output_path (str, optional): Path for the output DOCX file.
            If not provided, will use the same name as HTML file with .docx extension

    Returns:
        str: Path to the created DOCX file
    """
    # Read HTML file
    with open(html_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    # If no output path specified, create one based on input file
    if output_path is None:
        output_path = str(Path(html_path).with_suffix(".docx"))

    # Convert HTML to DOCX
    html2docx(html_content, output_path)

    return output_path
