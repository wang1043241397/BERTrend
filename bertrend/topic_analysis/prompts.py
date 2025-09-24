#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

TOPIC_DESCRIPTION_PROMPT_EN = """
You are a topic analysis expert. Your task is to generate a clear, human-readable title and a detailed description for the given topic.

Input:
- Topic representation: {topic_representation}
- Context documents (may include excerpts, keywords, or summaries):
{docs_text}

Requirements:
1) Title:
   - Be concise (max 8 words)
   - Be informative and specific
   - Use Title Case
   - Avoid jargon and buzzwords unless present in the context

2) Description:
   - ~100 words (90–120 words)
   - Summarize the central idea and scope
   - Highlight key themes, entities, or recurring concepts from the context
   - Avoid copying long verbatim text from the context; paraphrase instead
   - Be neutral, factual, and free of speculation
   - Do not begin with generic phrases such as "This theme...", "This topic...", or "The theme..."

3) Style & Quality:
   - No placeholders, no extraneous commentary
   - No first-person voice
   - No references to the prompt or instructions
   - Start directly with substantive content (e.g., a key concept, scope, or claim), not with meta-introductions

Output format (strict JSON; no prose outside JSON):
"title": "Your title here",  
"description": "Your ~100-word description here."
"""

TOPIC_DESCRIPTION_PROMPT_FR = """
Vous êtes un expert en analyse de sujets. Votre tâche est de générer un titre clair et lisible ainsi qu’une description détaillée pour le sujet fourni.

Entrée :
- Représentation du sujet : {topic_representation}
- Contexte (extraits, mots-clés ou résumés) :
{docs_text}

Exigences :
1) Titre :
   - Concis (max. 8 mots)
   - Informatif et spécifique
   - Éviter le jargon sauf s’il apparaît dans le contexte

2) Description :
   - Environ 100 mots (90–120)
   - Résumer l’idée centrale et le périmètre
   - Mettre en avant les thèmes, entités ou concepts récurrents du contexte
   - Ne pas copier de longs passages ; reformuler
   - Ton neutre, factuel, sans spéculation
   - Ne pas commencer par des formules génériques telles que « Ce thème… », « Ce sujet… » ou « Le thème… »

3) Style & Qualité :
   - Pas d’espaces réservés, pas de commentaires superflus
   - Pas de première personne
   - Aucune référence aux instructions
   - Commencer directement par le contenu substantiel (concept clé, périmètre, ou idée centrale), sans méta-introduction

Format de sortie (JSON strict ; aucun texte hors du JSON) :
"title": "Votre titre ici",  
"description": "Votre description (~100 mots) ici."
"""

TOPIC_DESCRIPTION_PROMPT = {
    "en": TOPIC_DESCRIPTION_PROMPT_EN,
    "fr": TOPIC_DESCRIPTION_PROMPT_FR,
}
