#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from pydantic import BaseModel


class TopicSummary(BaseModel):
    """Description of topic during a time period"""

    # Title of the topic
    title: str
    # Relevant date
    date: str
    # Key Developments: major developments or trends
    key_developments: list[str]
    # Analysis
    description: str
    # Novelty compared to previous period
    novelty: str


class TopicSummaryList(BaseModel):
    """Description of topic during a set of time period"""

    topic_summary_by_time_period: list[TopicSummary]


class PotentialImplications(BaseModel):
    """Potential Impact Analysis:
    - Examine the potential effects of this signal on various sectors, industries, and societal aspects.
    - Consider both short-term and long-term implications.
    - Analyze possible ripple effects and second-order consequences.
    """

    long_term_implications: list[str]
    short_term_implications: list[str]


class EvolutionScenario(BaseModel):
    """Evolution Scenarios
    - Describe potential ways this signal could develop or manifest in the future.
    - Consider various factors that could influence its trajectory.
    - Explore both optimistic and pessimistic scenarios.
    """

    optimistic_scenario_description: str
    optimistic_scenario_points: list[str]
    pessimistic_scenario_description: str
    pessimistic_scenario_points: list[str]


class TopicInterconnexions(BaseModel):
    """Interconnections and Synergies"""

    #  how this signal might interact with other current trends or emerging phenomena.
    interconnexions: list[str]
    #  potential synergies or conflicts with existing systems or paradigms.
    ripple_effects: list[str]


class Drivers(BaseModel):
    """Drivers and inhibitors"""

    # Factors that could accelerate or amplify a signal
    drivers: list[str]
    # Potential barriers or resistances that might hinder its development.
    inhibitors: list[str]


class SignalAnalysis(BaseModel):
    """Detailed analysis of topic evolution at a given time"""

    potential_implications: PotentialImplications
    evolution_scenario: EvolutionScenario
    topic_interconnexions: TopicInterconnexions
    drivers_inhibitors: Drivers
