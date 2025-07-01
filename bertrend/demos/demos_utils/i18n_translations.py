#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

# Translation dictionaries for demos_utils
TRANSLATIONS = {
    "ctrl_enter": {
        "fr": "CTRL + Entrée pour mettre à jour",
        "en": "CTRL + Enter to update",
    },
    "select_language": {"fr": "Choisir une langue", "en": "Select Language"},
    "embedding_model": {"fr": "Modèle d'embedding", "en": "Embedding Model"},
    "embedding_service_url": {
        "fr": "URL du service d'embedding",
        "en": "Embedding service URL",
    },
    "embedding_hyperparameters": {
        "fr": "Paramètres d'embedding",
        "en": "Embedding settings",
    },
    "embedding_service": {
        "fr": "Service d'embedding",
        "en": "Embedding Service",
    },
    "bertopic_hyperparameters": {
        "fr": "Hyperparamètres BERTopic",
        "en": "BERTopic Hyperparameters",
    },
    "bertrend_hyperparameters": {
        "fr": "Hyperparamètres BERTrend",
        "en": "BERTrend Hyperparameters",
    },
    "embeddings_calculated_message": {
        "fr": "Embeddings calculés avec succès !",
        "en": "Embeddings calculated successfully!",
    },
    "no_embeddings_warning_message": {
        "fr": "Veuillez vectoriser les données et entraîner les modèles avant de procéder à l'analyse.",
        "en": "Please embed data and train models before proceeding to analysis.",
    },
    "model_training_complete_message": {
        "fr": "Entraînement du modèle terminé !",
        "en": "Model training complete!",
    },
    "no_data_after_preprocessing_message": {
        "fr": "Aucune donnée disponible après prétraitement. Veuillez vérifier les fichiers sélectionnés et les options de prétraitement.",
        "en": "No data available after preprocessing. Please check the selected files and preprocessing options.",
    },
    "select_from_local_storage": {
        "fr": "Selection de jeux de données à partir du stockage local (.xlsx, .csv, .json, .jsonl, .parquet)",
        "en": "Select dataset from local storage (.xlsx, .csv, .json, .jsonl, .parquet)",
    },
    "select_from_remote_storage": {
        "fr": "Selection de jeux de données sur le serveur",
        "en": "Select one or more datasets from the server data",
    },
    "data_loading": {
        "fr": "Chargement des données",
        "en": "Data loading",
    },
    "local_data": {
        "fr": "Données locales",
        "en": "Data from local storage",
    },
    "remote_data": {"fr": "Données sur le serveur", "en": "Data from server"},
    "data_filtering": {"fr": "Filtrage des données", "en": "Data filtering"},
    "embed_documents": {
        "fr": "Vectoriser les documents",
        "en": "Embed Documents",
    },
    "embedding_documents": {
        "fr": "Vectorisation des documents...",
        "en": "Embedding documents...",
    },
    "no_dataset_warning": {
        "fr": "Veuillez sélectionner au moins un jeu de données pour continuer.",
        "en": "Please select at least one dataset to proceed.",
    },
    "error_loading_file": {
        "fr": "Erreur lors du chargement du fichier '{file_name}': {error}",
        "en": "Error while loading file '{file_name}': {error}",
    },
    "drag_drop_help": {
        "fr": "Glissez et déposez les fichiers à utiliser comme jeu de données dans cette zone",
        "en": "Drag and drop files to be used as dataset in this area",
    },
    "raw_documents_count": {
        "fr": "Nombre de documents dans les données brutes: **{count}**",
        "en": "Number of documents in raw data: **{count}**",
    },
    "minimum_characters": {
        "fr": "Nombre minimum de caractères",
        "en": "Minimum Characters",
    },
    "minimum_characters_help": {
        "fr": "Nombre minimum de caractères que chaque document doit contenir.",
        "en": "Minimum number of characters each document must contain.",
    },
    "sample_ratio": {
        "fr": "Ratio d'échantillonnage",
        "en": "Sample ratio",
    },
    "sample_ratio_help": {
        "fr": "Fraction des données brutes à utiliser pour calculer les sujets. Échantillonne aléatoirement les documents à partir des données brutes.",
        "en": "Fraction of raw data to use for computing topics. Randomly samples documents from raw data.",
    },
    "split_by_paragraphs": {
        "fr": "Diviser le texte par paragraphes",
        "en": "Split text by paragraphs",
    },
    "split_option_yes": {
        "fr": "oui",
        "en": "yes",
    },
    "split_option_no": {
        "fr": "non",
        "en": "no",
    },
    "split_option_enhanced": {
        "fr": "amélioré",
        "en": "enhanced",
    },
    "split_help": {
        "fr": "'Pas de division': Pas de division sur les documents ; 'Division par paragraphes': Divise les documents en paragraphes ; 'Division améliorée': utilise une méthode plus avancée mais plus lente pour la division qui prend en compte la longueur d'entrée maximale du modèle d'embedding.",
        "en": "'No split': No splitting on the documents ; 'Split by paragraphs': Split documents into paragraphs ; 'Enhanced split': uses a more advanced but slower method for splitting that considers the embedding model's maximum input length.",
    },
    "select_timeframe": {
        "fr": "Sélectionner la période",
        "en": "Select Timeframe",
    },
    "filtered_documents_count": {
        "fr": "Nombre de documents dans les données filtrées: **{count}**",
        "en": "Number of documents in filtered data: **{count}**",
    },
    "settings_and_controls": {
        "fr": "Paramètres et contrôles",
        "en": "Settings and Controls",
    },
    "column_selection": {
        "fr": "Sélection des colonnes",
        "en": "Column Selection",
    },
    "text_column_selection": {
        "fr": "Sélection de la colonne contenant le texte",
        "en": "Select Column Containing Text",
    },
    "timestamp_column_selection": {
        "fr": "Sélection de la colonne contenant l'horodatage",
        "en": "Select Column Containing Timestamp",
    },
}
