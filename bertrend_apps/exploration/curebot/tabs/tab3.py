import locale
import jinja2
import streamlit as st
from bertrend_apps.exploration.curebot.app_utils import (
    NEWSLETTER_TEMPLATE,
    create_newsletter,
)
from bertrend_apps.exploration.curebot.prompts import TOPIC_SUMMARY_SYSTEM_PROMPT

# Set french locale
locale.setlocale(locale.LC_ALL, "fr_FR.UTF-8")


def show() -> None:
    if "topic_model" not in st.session_state:
        st.warning("Veuillez ajouter des données et lancer la détection des sujets.")
    else:
        # Show newsletter parameters
        newsletter_parameters()

        # Show newsletter creation button
        if st.button("Créer la newsletter"):
            with st.spinner("Création de la newsletter..."):
                get_newsletter()

        # If newsletter is created, show it and add downlaod button
        if "newsletter" in st.session_state:
            template = jinja2.Template(NEWSLETTER_TEMPLATE)
            st.write(template.render(st.session_state["newsletter"]))


def newsletter_parameters() -> None:
    """
    Show newsletter parameters:
    - Number of topics to include in the newsletter
    - Number of articles per topic to include in the newsletter
    """
    col31, col32 = st.columns([1, 1])
    with col31:
        # Select number of topics to include in the newsletter
        st.slider(
            "Nombre de sujets",
            1,
            len(st.session_state["topics_info"]),
            value=min(len(st.session_state["topics"]), 3),
            key="newsletter_nb_topics",
        )
    with col32:
        # Select number of articles per topic to include in the newsletter
        st.slider(
            "Nombre d'articles par sujet",
            1,
            10,
            value=4,
            key="newsletter_nb_articles_per_topic",
        )


def get_newsletter() -> None:
    """
    Create a newsletter based on the selected topics and articles.
    Sets in sessions_state:
    - newsletter: a dictionary containing the newsletter data
    """
    st.session_state["newsletter"] = create_newsletter(
        st.session_state["df"],
        st.session_state["topics_info"],
        st.session_state["newsletter_nb_topics"],
        st.session_state["newsletter_nb_articles_per_topic"],
    )
