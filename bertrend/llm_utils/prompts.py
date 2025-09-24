#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

FR_USER_SUMMARY_MULTIPLE_DOCS = """
Vous êtes une IA spécialisée dans la compréhension et la synthèse d’articles de presse.
Votre tâche : produire une synthèse concise et fidèle au contenu fourni.

Entrées :
- Mots-clés du thème : {keywords}
- Articles (Titre + Contenu) :
```{article_list}```

Exigences de la synthèse :
1) Contenu :
   - Être directement liée au thème évoqué par les mots-clés.
   - Intégrer les informations essentielles communes ou complémentaires entre les articles.
   - Éviter les répétitions, les détails anecdotiques et le bruit.
   - Mentionner les acteurs, faits, dates, chiffres et positions clés quand disponibles.
   - Être neutre, factuelle, sans spéculation ni jugement.
2) Forme :
   - Ne pas dépasser {nb_sentences} phrases (phrases complètes, séparées par des points).
   - Ne pas commencer par « Les articles… », « Ces articles… » ni toute méta-introduction.
   - Commencer directement par l’information la plus saillante.
   - Pas de listes, ni de puces ; texte continu.
   - Pas de citations longues ; reformuler brièvement si nécessaire.
3) Qualité & Cohérence :
   - Résoudre les divergences entre articles en signalant la divergence de façon concise si pertinente.
   - Éviter les doublons d’information.
   - Ne pas inventer d’informations absentes de l’entrée.
   - Respecter la langue des consignes (français).

Sortie attendue :
- Un seul paragraphe, {nb_sentences} phrases maximum.
- Aucune introduction, aucune conclusion générique, aucun texte en dehors de la synthèse.
"""

EN_USER_SUMMARY_MULTIPLE_DOCS = """
You are an AI specialized in understanding and synthesizing news articles.
Your task: produce a concise and faithful summary of the provided content.

Inputs:
- Topic keywords: {keywords}
- Articles (Title + Content):
```{article_list}```

Summary requirements:
1) Content:
   - Be directly related to the topic evoked by the keywords.
   - Capture the essential, shared, or complementary information across articles.
   - Avoid repetition, minor details, and noise.
   - Include key actors, facts, dates, figures, and stated positions when available.
   - Maintain a neutral, factual tone with no speculation or judgment.
2) Form:
   - Do not exceed {nb_sentences} sentences (complete sentences, period-separated).
   - Do not begin with “The articles…”, “These articles…”, or any meta-introduction.
   - Start immediately with the most salient information.
   - No lists or bullets; continuous prose only.
   - No long quotations; paraphrase briefly if needed.
3) Quality & Consistency:
   - If articles diverge, briefly note the discrepancy only if relevant.
   - Avoid duplicating the same information.
   - Do not invent information not present in the input.
   - Output in English.

Expected output:
- A single paragraph, at most {nb_sentences} sentences.
- No introduction, no generic conclusion, and no text outside the summary.
"""

# keywords: list of keywords describing the topic
# article_list: list of articles (each with title and content)
USER_SUMMARY_MULTIPLE_DOCS = {
    "fr": FR_USER_SUMMARY_MULTIPLE_DOCS,
    "en": EN_USER_SUMMARY_MULTIPLE_DOCS,
}

###################### TOPIC PROMPTS

FR_USER_GENERATE_TOPIC_LABEL_SUMMARIES = """
Contexte : génération d’une newsletter sur le thème « {newsletter_title} ».
Tâche : proposer un libellé très court décrivant le sous-thème spécifique du texte fourni.

Contraintes :
- Longueur : au plus 5 mots.
- Clarté : formulation naturelle, lisible, sans jargon inutile.
- Spécificité : refléter ce qui rend le texte distinctif dans le cadre du thème « {newsletter_title} ».
- Éviter :
  - Libellés trop généraux (ex. : « Actualités », « Mise à jour », « Tendances »).
  - Libellés trop proches ou quasi identiques au thème principal (ex. répéter « {newsletter_title} » ou son équivalent).
  - Formulations métadiscursives (« Ce texte… », « Sous-thème… »).
  - Ponctuation superflue (pas de point final si possible).
- Langue : français.
- Un seul libellé attendu.

Texte :
"{title_list}"

Sortie (uniquement le libellé, sans guillemets, sans texte additionnel) :
"""

EN_USER_GENERATE_TOPIC_LABEL_SUMMARIES = """
Context: generating a newsletter on the topic "{newsletter_title}".
Task: propose a very short label that captures the specific sub-topic of the provided text.

Constraints:
- Length: maximum 5 words.
- Clarity: natural, readable phrasing; avoid unnecessary jargon.
- Specificity: reflect what makes the text distinctive within the scope of "{newsletter_title}".
- Avoid:
  - Overly general labels (e.g., "News", "Update", "Trends").
  - Labels too close to or nearly identical to the main topic (e.g., repeating "{newsletter_title}" or its near-synonym).
  - Meta-introductions ("This text…", "Subtopic…").
  - Superfluous punctuation (no trailing period if possible).
- Language: English.
- Exactly one label expected.

Text:
"{title_list}"

Output (only the label, no quotes, no extra text):
"""

# title_list: list or block of document excerpts belonging to the topic
USER_GENERATE_TOPIC_LABEL_SUMMARIES = {
    "fr": FR_USER_GENERATE_TOPIC_LABEL_SUMMARIES,
    "en": EN_USER_GENERATE_TOPIC_LABEL_SUMMARIES,
}

BERTOPIC_FRENCH_TOPIC_REPRESENTATION_PROMPT = (
    "J'ai un topic qui contient les documents suivants :\n"
    "[DOCUMENTS]\n"
    "Le topic est décrit par les mots-clés suivants : [KEYWORDS]\n"
    "Sur la base des informations ci-dessus, extraire une courte étiquette de topic dans le format suivant :\n"
    "Topic : <étiquette du sujet>"
)
# Passed directly to BERTopic's OpenAI wrapper, formatted similarly to BERTopic's original prompt which can be found in its source code
