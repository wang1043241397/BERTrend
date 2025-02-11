import streamlit as st

from bertrend_apps.exploration.curebot.tabs import tab1, tab2, tab3

# Set wide layout
st.set_page_config(
    page_title="Curebot - Exploration de sujets",
    layout="wide",
)

# Set app title
st.title("Curebot - Exploration de sujets")

# Set sidebar
with st.sidebar:
    st.header("BERTopic")
    with st.expander("Paramètres"):
        st.checkbox("Utiliser les tags", key="use_tags", value=False)
        st.slider(
            "Nombre d'articles minimum par sujet",
            1,
            50,
            10,
            key="min_articles_per_topic",
        )

# Create tabs
tab1_content, tab2_content, tab3_content = st.tabs(
    ["Données", "Résultats", "Newsletter"]
)

# Main tabs
with tab1_content:
    tab1.show()

with tab2_content:
    tab2.show()

with tab3_content:
    tab3.show()
