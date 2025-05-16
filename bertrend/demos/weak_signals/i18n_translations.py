#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

# Translation dictionaries for weak_signals demo
TRANSLATIONS = {
    # app.py translations
    "page_title": {
        "fr": "BERTrend - Démo d'analyse rétrospective de tendances",
        "en": "BERTrend - Trend Retrospective Analysis demo",
    },
    "data_loading": {
        "fr": "Chargement des données",
        "en": "Data Loading",
    },
    "model_training": {
        "fr": "Entraînement du modèle",
        "en": "Model Training",
    },
    "results_analysis": {
        "fr": "Analyse des résultats",
        "en": "Results Analysis",
    },
    "state_management": {
        "fr": "Gestion de l'état",
        "en": "State Management",
    },
    "restore_previous_run": {
        "fr": "Restaurer l'exécution précédente",
        "en": "Restore Previous Run",
    },
    "purge_cache": {
        "fr": "Purger le cache",
        "en": "Purge Cache",
    },
    "clear_session_state": {
        "fr": "Effacer l'état de la session",
        "en": "Clear session state",
    },
    "data_loading_and_preprocessing": {
        "fr": "Chargement et prétraitement des données",
        "en": "Data Loading and Preprocessing",
    },
    "documents_per_timestamp": {
        "fr": "Documents par horodatage",
        "en": "Documents per Timestamp",
    },
    "granularity": {
        "fr": "Granularité",
        "en": "Granularity",
    },
    "select_timestamp": {
        "fr": "Sélectionner l'horodatage",
        "en": "Select Timestamp",
    },
    "enter_zeroshot_topics": {
        "fr": "Entrez les sujets zero-shot (séparés par /)",
        "en": "Enter zero-shot topics (separated by /)",
    },
    "train_models": {
        "fr": "Entraîner les modèles",
        "en": "Train Models",
    },
    "training_models": {
        "fr": "Entraînement des modèles...",
        "en": "Training models...",
    },
    "topic_overview": {
        "fr": "Aperçu des sujets",
        "en": "Topic Overview",
    },
    "zeroshot_weak_signal_trends": {
        "fr": "Tendances des signaux faibles zero-shot",
        "en": "Zero-shot Weak Signal Trends",
    },
    "popularity_of_zeroshot_topics": {
        "fr": "Popularité des sujets zero-shot",
        "en": "Popularity of Zero-Shot Topics",
    },
    "timestamp": {
        "fr": "Horodatage",
        "en": "Timestamp",
    },
    "popularity": {
        "fr": "Popularité",
        "en": "Popularity",
    },
    "zeroshot_topics_data_saved": {
        "fr": "Données des sujets zero-shot sauvegardées dans {json_file_path}",
        "en": "Zeroshot topics data saved to {json_file_path}",
    },
    "topic_size_evolution": {
        "fr": "Évolution de la taille des sujets",
        "en": "Topic Size Evolution",
    },
    "topic_popularity_evolution": {
        "fr": "Évolution de la popularité des sujets",
        "en": "Topic Popularity Evolution",
    },
    "signal_analysis": {
        "fr": "Analyse du signal",
        "en": "Signal Analysis",
    },
    "enter_topic_number": {
        "fr": "Entrez un numéro de sujet pour l'examiner de plus près :",
        "en": "Enter a topic number to take a closer look:",
    },
    "analyze_signal": {
        "fr": "Analyser le signal",
        "en": "Analyze signal",
    },
    "error_generating_signal_summary": {
        "fr": "Erreur lors de la génération du résumé du signal : {e}",
        "en": "Error while trying to generate signal summary: {e}",
    },
    "error_embedding_documents": {
        "fr": "Une erreur s'est produite lors de la vectorisation : {e}",
        "en": "An error occurred while embedding documents: {e}",
    },
    "topic_evolution": {
        "fr": "Évolution des sujets",
        "en": "Topic Evolution",
    },
    "newly_emerged_topics": {
        "fr": "Sujets nouvellement apparus",
        "en": "Newly Emerged Topics",
    },
    "retrieve_topic_counts": {
        "fr": "Récupérer les comptages de sujets",
        "en": "Retrieve Topic Counts",
    },
    "retrieving_topic_counts": {
        "fr": "Récupération des comptages de sujets...",
        "en": "Retrieving topic counts...",
    },
    "state_saved_message": {
        "fr": "État de l'application sauvegardé.",
        "en": "Application state saved.",
    },
    "state_restored_message": {
        "fr": "État de l'application restauré.",
        "en": "Application state restored.",
    },
    "models_saved_message": {
        "fr": "Modèles sauvegardés.",
        "en": "Models saved.",
    },
    "models_restored_message": {
        "fr": "Modèles restaurés.",
        "en": "Models restored.",
    },
    "model_merging_complete_message": {
        "fr": "Fusion des modèles terminée !",
        "en": "Model merging complete!",
    },
    "topic_counts_saved_message": {
        "fr": "Comptages de sujets et de signaux sauvegardés dans {file_path}",
        "en": "Topic and signal counts saved to {file_path}",
    },
    "cache_purged_message": {
        "fr": "Cache purgé.",
        "en": "Cache purged.",
    },
    "progress_bar_description": {
        "fr": "Lots traités",
        "en": "Batches processed",
    },
    "no_data_warning": {
        "fr": "Aucune donnée disponible pour la granularité sélectionnée.",
        "en": "No data available for the selected granularity.",
    },
    "no_state_warning": {
        "fr": "Aucun état sauvegardé trouvé.",
        "en": "No saved state found.",
    },
    "no_models_warning": {
        "fr": "Aucun modèle sauvegardé trouvé.",
        "en": "No saved models found.",
    },
    "no_cache_warning": {
        "fr": "Aucun cache trouvé.",
        "en": "No cache found.",
    },
    "embed_warning": {
        "fr": "Veuillez vectoriser les données avant de procéder à l'entraînement du modèle.",
        "en": "Please embed data before proceeding to model training.",
    },
    "embed_train_warning": {
        "fr": "Veuillez vectoriser les données et entraîner les modèles avant de procéder à l'analyse.",
        "en": "Please embed data and train models before proceeding to analysis.",
    },
    "train_warning": {
        "fr": "Veuillez entraîner les modèles avant de procéder à l'analyse.",
        "en": "Please train models before proceeding to analysis.",
    },
    "merge_warning": {
        "fr": "Veuillez fusionner les modèles pour voir des analyses supplémentaires.",
        "en": "Please merge models to view additional analyses.",
    },
    "topic_not_found_warning": {
        "fr": "Le sujet {topic_number} n'a pas été trouvé dans l'historique des fusions dans la fenêtre spécifiée.",
        "en": "Topic {topic_number} not found in the merge histories within the specified window.",
    },
    "no_granularity_warning": {
        "fr": "Valeur de granularité non trouvée.",
        "en": "Granularity value not found.",
    },
    "html_generation_failed_warning": {
        "fr": "La génération HTML a échoué. Affichage du markdown à la place.",
        "en": "HTML generation failed. Displaying markdown instead.",
    },
    "search_topics_by_keywords": {
        "en": "Search topics by keyword:",
        "fr": "Recherche des sujets par mots-clés :",
    },
    "retrospective_period": {
        "en": "Retrospective Period (days)",
        "fr": "Période rétrospective (jours)",
    },
    "current_date": {"en": "Current date", "fr": "Date actuelle"},
    "current_date_help": {
        "en": "The earliest selectable date corresponds to the earliest timestamp when topics were merged (with the smallest possible value being the earliest timestamp in the provided data). The latest selectable date corresponds to the most recent topic merges, which is at most equal to the latest timestamp in the data minus the provided granularity.",
        "fr": "La date sélectionnable la plus ancienne correspond au premier horodatage où des sujets ont été fusionnés (avec la valeur minimale possible étant le premier horodatage dans les données fournies). La date sélectionnable la plus récente correspond aux fusions de sujets les plus récentes, qui est au plus égale au dernier horodatage dans les données moins la granularité fournie.",
    },
    "noise_threshold": {
        "en": "Noise Threshold : {value}",
        "fr": "Seuil de bruit : {value}",
    },
    "strong_signal_threshold": {
        "en": "Strong Signal Threshold : {value}",
        "fr": "Seuil de signal fort : {value}",
    },
    "weak_signals": {"en": "Weak Signals", "fr": "Signaux faibles"},
    "strong_signals": {"en": "Strong Signals", "fr": "Signaux forts"},
    "noise": {"en": "Noise", "fr": "Bruit"},
    "no_weak_signals": {
        "en": "No weak signals were detected at timestamp {timestamp}.",
        "fr": "Aucun signal faible n'a été détecté à l'horodate {timestamp}.",
    },
    "no_strong_signals": {
        "en": "No strong signals were detected at timestamp {timestamp}.",
        "fr": "Aucun signal fort n'a été détecté à l'horodate {timestamp}.",
    },
    "no_noise_signals": {
        "en": "No noisy signals were detected at timestamp {timestamp}.",
        "fr": "Aucun signal de bruit n'a été détecté à l'horodate {timestamp}.",
    },
    "topic_merging_process": {
        "en": "Topic Merging Process",
        "fr": "Processus de fusion des sujets",
    },
    "max_topic_pairs": {
        "en": "Max number of topic pairs to display",
        "fr": "Nombre maximum de paires de sujets à afficher",
    },
    "explore_topic_models": {
        "en": "Explore topic models",
        "fr": "Explorer les modèles de sujets",
    },
    "select_model": {"en": "Select Model", "fr": "Sélectionner le modèle"},
    "signal_interpretation": {
        "en": "Signal Interpretation",
        "fr": "Interprétation du signal",
    },
    "analyzing_signal": {"en": "Analyzing signal...", "fr": "Analyse du signal..."},
    "select_date_range": {
        "en": "Select date range for saving signal evolution data:",
        "fr": "Sélectionnez la plage de dates pour l'enregistrement des données d'évolution du signal :",
    },
    "save_signal_evolution_data": {
        "en": "Save Signal Evolution Data",
        "fr": "Enregistrer les données d'évolution du signal",
    },
    "data_saved_success": {
        "en": "Signal evolution data saved successfully at {path}",
        "fr": "Données d'évolution du signal enregistrées avec succès à {path}",
    },
    "data_saved_error": {
        "en": "Error encountered while saving signal evolution data: {error}",
        "fr": "Erreur rencontrée lors de l'enregistrement des données d'évolution du signal : {error}",
    },
    "topic_counts_saved": {
        "en": "Topic counts for individual and cumulative merged models saved to {path}",
        "fr": "Nombre de sujets pour les modèles individuels et fusionnés cumulatifs enregistrés dans {path}",
    },
}
