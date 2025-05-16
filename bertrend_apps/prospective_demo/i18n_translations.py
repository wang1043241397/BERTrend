#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

# Translation dictionaries for prospective_demo
TRANSLATIONS = {
    # App.py translations
    "app_title": {
        "fr": "BERTrend - Démo Veille & Analyse",
        "en": "BERTrend - Monitoring & Analysis Demo",
    },
    "tab_monitoring": {"fr": "Veilles", "en": "Monitoring"},
    "tab_models": {"fr": "Modèles", "en": "Models"},
    "tab_trends": {"fr": "Tendances", "en": "Trends"},
    "tab_analysis": {"fr": "Analyses", "en": "Analysis"},
    "tab_reports": {"fr": "Génération de rapports", "en": "Report Generation"},
    "data_flow_config": {
        "fr": "Configuration des flux de données",
        "en": "Data Flow Configuration",
    },
    "data_collection_status": {
        "fr": "Etat de collecte des données",
        "en": "Data Collection Status",
    },
    "model_status_by_monitoring": {
        "fr": "Statut des modèles par veille",
        "en": "Model Status by Monitoring",
    },
    # dashboard_analysis.py translations
    "detailed_analysis_by_topic": {
        "fr": "Analyse détaillée par sujet",
        "en": "Detailed Analysis by Topic",
    },
    "topic_selection": {"fr": "Sélection du sujet", "en": "Topic Selection"},
    "emerging_topic": {"fr": "Sujet émergent", "en": "Emerging Topic"},
    "strong_topic": {"fr": "Sujet fort", "en": "Strong Topic"},
    "nothing_to_display": {"fr": "Rien à afficher", "en": "Nothing to display"},
    # dashboard_signals.py translations
    "title": {"fr": "Titre", "en": "Title"},
    "todo_message": {
        "fr": "TODO",
        "en": "TODO",
    },
    "explore_sources_by_topic": {
        "fr": "Exploration des sources par sujet",
        "en": "Explore Sources by Topic",
    },
    "signal_type": {"fr": "Type de signal", "en": "Signal Type"},
    "emerging_topics": {"fr": "Sujets émergents", "en": "Emerging Topics"},
    "strong_topics": {"fr": "Sujets forts", "en": "Strong Topics"},
    "no_data": {"fr": "Pas de données", "en": "No data"},
    "topic": {"fr": "Sujet", "en": "Topic"},
    "untitled_topic": {"fr": "???Titre???", "en": "???Title???"},
    "explore_sources": {"fr": "Exploration des sources", "en": "Explore Sources"},
    "reference_articles": {"fr": "Articles de référence", "en": "Reference Articles"},
    # Signal categories
    "weak_signals": {"fr": "Signaux faibles", "en": "Weak Signals"},
    "strong_signals": {"fr": "Signaux forts", "en": "Strong Signals"},
    "noise": {"fr": "Bruit", "en": "Noise"},
    "no_weak_signals": {
        "fr": "Aucun signal faible n'a été détecté à l'horodatage {timestamp}.",
        "en": "No weak signals were detected at timestamp {timestamp}.",
    },
    "no_strong_signals": {
        "fr": "Aucun signal fort n'a été détecté à l'horodatage {timestamp}.",
        "en": "No strong signals were detected at timestamp {timestamp}.",
    },
    "no_noise_signals": {
        "fr": "Aucun signal de bruit n'a été détecté à l'horodatage {timestamp}.",
        "en": "No noisy signals were detected at timestamp {timestamp}.",
    },
}
