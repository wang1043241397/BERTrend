#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

"""
Comparative analysis dashboard for comparing topic trends between different time periods.

This module provides functionality to:
- Select and compare two different time periods
- Analyze topic evolution (new, disappeared, stable topics)
- Compare popularity changes across topics
- Visualize source diversity changes
- Display comparative metrics with interactive charts
"""

from datetime import datetime
from typing import Any
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from bertrend.demos.demos_utils.icons import WARNING_ICON, ANALYSIS_ICON, ERROR_ICON
from bertrend.demos.demos_utils.i18n import translate
from bertrend_apps.prospective_demo import (
    NOISE,
    WEAK_SIGNALS,
    STRONG_SIGNALS,
    LLM_TOPIC_TITLE_COLUMN,
    get_model_interpretation_path,
)
from bertrend_apps.prospective_demo.dashboard_common import (
    get_df_topics,
    update_key,
)
from bertrend_apps.prospective_demo.models_info import get_models_info

# Constants
DEFAULT_CHART_HEIGHT = 400
TOP_TOPICS_DISPLAY_LIMIT = 10
DATE_FORMAT = "%d/%m/%Y"
COLUMN_TOPIC = "Topic"
COLUMN_POPULARITY = "Latest_Popularity"
COLUMN_SOURCE_DIVERSITY = "Source_Diversity"
SIGNAL_CATEGORIES = [WEAK_SIGNALS, STRONG_SIGNALS]


def choose_two_periods() -> tuple[str, datetime, datetime]:
    """
    Allow user to select two different time periods for comparison.
    
    This function displays UI controls for selecting a feed and two analysis periods.
    It validates that at least 2 periods are available before allowing selection.
    
    Returns:
        tuple[str, datetime, datetime]: A tuple containing:
            - model_id: The selected feed/model identifier
            - period_1: The first selected analysis period
            - period_2: The second selected analysis period
    
    Raises:
        st.stop(): If no models are available or fewer than 2 periods exist
    """
    col1, col2 = st.columns(2)
    
    # Get available models/timestamps with validation
    try:
        options = sorted(st.session_state.user_feeds.keys())
        if not options:
            st.error(translate("no_available_model_warning"), icon=ERROR_ICON)
            st.stop()
    except (AttributeError, KeyError) as e:
        st.error(f"Error accessing user feeds: {e}", icon=ERROR_ICON)
        st.stop()
    
    # Initialize model_id in session state if not present
    if "model_id" not in st.session_state:
        st.session_state.model_id = options[0]
    
    # Model selection
    model_id = st.selectbox(
        translate("select_feed"),
        options=options,
        index=options.index(st.session_state.model_id),
        key="comparative_model_id",
    )
    st.session_state.model_id = model_id
    
    # Get available periods for selected model
    try:
        list_models = get_models_info(model_id, st.session_state.username)
    except Exception as e:
        st.error(f"Error loading model information: {e}", icon=ERROR_ICON)
        st.stop()
    
    if not list_models:
        st.warning(translate("no_available_model_warning"), icon=WARNING_ICON)
        st.stop()
    elif len(list_models) < 2:
        st.warning(translate("at_least_2models_warning"), icon=WARNING_ICON)
        st.stop()
    
    # Period 1 selection
    with col1:
        period1_key = uuid.uuid4()
        if "period_1" not in st.session_state:
            # Default to second-to-last period if available
            st.session_state.period_1 = list_models[-2] if len(list_models) >= 2 else list_models[0]
        period_1 = st.selectbox(
            translate("period_1"),
            options=list_models,
            index=list_models.index(st.session_state.period_1),
            format_func=lambda ts: ts.strftime(DATE_FORMAT),
            key=period1_key,
            on_change=lambda: update_key("period_1", st.session_state[period1_key]),
        )
    
    # Period 2 selection
    with col2:
        period2_key = uuid.uuid4()
        if "period_2" not in st.session_state:
            # Default to most recent period
            st.session_state.period_2 = list_models[-1]
        period_2 = st.selectbox(
            translate("period_2"),
            options=list_models,
            index=list_models.index(st.session_state.period_2),
            format_func=lambda ts: ts.strftime(DATE_FORMAT),
            key=period2_key,
            on_change=lambda: update_key("period_2", st.session_state[period2_key]),
        )
    
    return model_id, period_1, period_2


@st.cache_data(ttl=3600, show_spinner=False)
def get_period_data(model_id: str, period: datetime, username: str) -> dict[str, pd.DataFrame]:
    """
    Load topic data for a specific period with caching.
    
    This function retrieves topic data (weak signals, strong signals, noise) for a given
    model and time period. Results are cached for 1 hour to improve performance.
    
    Args:
        model_id: The feed/model identifier
        period: The analysis period timestamp
        username: The username for accessing user-specific data
    
    Returns:
        dict[str, pd.DataFrame]: Dictionary containing DataFrames for each signal type:
            - WEAK_SIGNALS: DataFrame of weak signal topics
            - STRONG_SIGNALS: DataFrame of strong signal topics
            - NOISE: DataFrame of noise topics
    
    Raises:
        Exception: If data cannot be loaded or path is invalid
    """
    try:
        model_interpretation_path = get_model_interpretation_path(
            user_name=username,
            model_id=model_id,
            reference_ts=period,
        )
        return get_df_topics(model_interpretation_path)
    except Exception as e:
        st.error(f"Error loading data for period {period.strftime(DATE_FORMAT)}: {e}", icon=ERROR_ICON)
        # Return empty dataframes to allow graceful degradation
        return {
            WEAK_SIGNALS: pd.DataFrame(),
            STRONG_SIGNALS: pd.DataFrame(),
            NOISE: pd.DataFrame(),
        }


def compare_topic_counts(
    data1: dict[str, pd.DataFrame], 
    data2: dict[str, pd.DataFrame], 
    period1: datetime, 
    period2: datetime
) -> None:
    """
    Compare the number of topics between two periods using a grouped bar chart.
    
    Displays a visualization showing the count of weak signals, strong signals, and noise
    for each period side by side.
    
    Args:
        data1: Dictionary of DataFrames for the first period
        data2: Dictionary of DataFrames for the second period
        period1: First period timestamp
        period2: Second period timestamp
    """
    st.subheader(translate("topic_count_comparison"))
    
    try:
        # Prepare data for visualization
        categories = [translate("weak_signals"), translate("strong_signals"), translate("noise")]
        period1_counts = [
            len(data1.get(WEAK_SIGNALS, pd.DataFrame())),
            len(data1.get(STRONG_SIGNALS, pd.DataFrame())),
            len(data1.get(NOISE, pd.DataFrame())),
        ]
        period2_counts = [
            len(data2.get(WEAK_SIGNALS, pd.DataFrame())),
            len(data2.get(STRONG_SIGNALS, pd.DataFrame())),
            len(data2.get(NOISE, pd.DataFrame())),
        ]
        
        # Create grouped bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=period1.strftime(DATE_FORMAT),
            x=categories,
            y=period1_counts,
            marker_color='lightblue',
            text=period1_counts,
            textposition='auto',
        ))
        fig.add_trace(go.Bar(
            name=period2.strftime(DATE_FORMAT),
            x=categories,
            y=period2_counts,
            marker_color='darkblue',
            text=period2_counts,
            textposition='auto',
        ))
        
        fig.update_layout(
            barmode='group',
            xaxis_title=translate("signal_type"),
            yaxis_title=translate("topic_count_comparison"),
            height=DEFAULT_CHART_HEIGHT,
            hovermode='x unified',
            showlegend=True,
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating topic count comparison: {e}", icon=ERROR_ICON)


def _extract_topic_ids(data: dict[str, pd.DataFrame]) -> set[int]:
    """
    Extract all topic IDs from weak and strong signal DataFrames.
    
    Args:
        data: Dictionary containing signal DataFrames
        
    Returns:
        set[int]: Set of unique topic IDs found in the data
    """
    topics = set()
    for category in SIGNAL_CATEGORIES:
        df = data.get(category, pd.DataFrame())
        if not df.empty and COLUMN_TOPIC in df.columns:
            topics.update(df[COLUMN_TOPIC].tolist())
    return topics


def analyze_topic_evolution(
    data1: dict[str, pd.DataFrame], 
    data2: dict[str, pd.DataFrame], 
    period1: datetime, 
    period2: datetime
) -> None:
    """
    Analyze topic evolution between two periods.
    
    Identifies and displays metrics for:
    - New topics that appeared in period 2
    - Topics that disappeared from period 1
    - Stable topics present in both periods
    
    Args:
        data1: Dictionary of DataFrames for the first period
        data2: Dictionary of DataFrames for the second period
        period1: First period timestamp
        period2: Second period timestamp
    """
    st.subheader(translate("signal_evolution"))
    
    try:
        # Extract topic IDs from both periods
        topics1 = _extract_topic_ids(data1)
        topics2 = _extract_topic_ids(data2)
        
        # Calculate evolution metrics
        new_topics = topics2 - topics1
        disappeared_topics = topics1 - topics2
        stable_topics = topics1 & topics2
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label=translate("new_topics"),
                value=len(new_topics),
                delta=f"+{len(new_topics)}",
            )
        
        with col2:
            st.metric(
                label=translate("disappeared_topics"),
                value=len(disappeared_topics),
                delta=f"-{len(disappeared_topics)}",
                delta_color="inverse",
            )
        
        with col3:
            st.metric(
                label=translate("stable_topics"),
                value=len(stable_topics),
            )
        
        # Show detailed information in expanders
        if new_topics:
            with st.expander(f"📈 {translate('new_topics')} ({len(new_topics)})"):
                display_topic_list(data2, list(new_topics))
        
        if disappeared_topics:
            with st.expander(f"📉 {translate('disappeared_topics')} ({len(disappeared_topics)})"):
                display_topic_list(data1, list(disappeared_topics))
        
        if stable_topics:
            with st.expander(f"🔄 {translate('stable_topics')} ({len(stable_topics)})"):
                compare_stable_topics(data1, data2, list(stable_topics), period1, period2)
    
    except Exception as e:
        st.error(f"Error analyzing topic evolution: {e}", icon=ERROR_ICON)


def display_topic_list(data: dict[str, pd.DataFrame], topic_ids: list[int]) -> None:
    """
    Display a list of topics with their titles and metadata in a table.
    
    Args:
        data: Dictionary of DataFrames containing topic information
        topic_ids: List of topic IDs to display
    """
    try:
        topics_info = []
        
        for category in SIGNAL_CATEGORIES:
            df = data.get(category, pd.DataFrame())
            if df.empty or COLUMN_TOPIC not in df.columns:
                continue
                
            for topic_id in topic_ids:
                topic_row = df[df[COLUMN_TOPIC] == topic_id]
                if not topic_row.empty:
                    title = (
                        topic_row[LLM_TOPIC_TITLE_COLUMN].values[0] 
                        if LLM_TOPIC_TITLE_COLUMN in topic_row.columns 
                        else "N/A"
                    )
                    popularity = (
                        topic_row[COLUMN_POPULARITY].values[0] 
                        if COLUMN_POPULARITY in topic_row.columns 
                        else 0
                    )
                    topics_info.append({
                        COLUMN_TOPIC: topic_id,
                        translate("title"): title,
                        "Popularity": popularity,
                        "Category": (
                            translate("weak_signals") 
                            if category == WEAK_SIGNALS 
                            else translate("strong_signals")
                        ),
                    })
        
        if topics_info:
            st.dataframe(pd.DataFrame(topics_info), use_container_width=True)
        else:
            st.info(translate("no_data"))
    
    except Exception as e:
        st.error(f"Error displaying topic list: {e}", icon=ERROR_ICON)


def compare_stable_topics(
    data1: dict[str, pd.DataFrame], 
    data2: dict[str, pd.DataFrame], 
    topic_ids: list[int], 
    period1: datetime, 
    period2: datetime
) -> None:
    """
    Compare popularity changes for topics that appear in both periods.
    
    Displays a table and bar chart showing how popularity changed for stable topics
    between the two periods.
    
    Args:
        data1: Dictionary of DataFrames for the first period
        data2: Dictionary of DataFrames for the second period
        topic_ids: List of topic IDs that appear in both periods
        period1: First period timestamp
        period2: Second period timestamp
    """
    try:
        comparison_data = []
        
        for category in SIGNAL_CATEGORIES:
            df1 = data1.get(category, pd.DataFrame())
            df2 = data2.get(category, pd.DataFrame())
            
            if df1.empty or df2.empty:
                continue
                
            for topic_id in topic_ids:
                topic1 = df1[df1[COLUMN_TOPIC] == topic_id]
                topic2 = df2[df2[COLUMN_TOPIC] == topic_id]
                
                if not topic1.empty and not topic2.empty:
                    title = (
                        topic2[LLM_TOPIC_TITLE_COLUMN].values[0] 
                        if LLM_TOPIC_TITLE_COLUMN in topic2.columns 
                        else f"Topic {topic_id}"
                    )
                    pop1 = (
                        topic1[COLUMN_POPULARITY].values[0] 
                        if COLUMN_POPULARITY in topic1.columns 
                        else 0
                    )
                    pop2 = (
                        topic2[COLUMN_POPULARITY].values[0] 
                        if COLUMN_POPULARITY in topic2.columns 
                        else 0
                    )
                    change = pop2 - pop1
                    change_percent = (change / pop1 * 100) if pop1 > 0 else 0
                    
                    comparison_data.append({
                        COLUMN_TOPIC: topic_id,
                        translate("title"): title,
                        period1.strftime(DATE_FORMAT): pop1,
                        period2.strftime(DATE_FORMAT): pop2,
                        translate("popularity_change"): change,
                        "Change %": change_percent,
                    })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            df = df.sort_values(by=translate("popularity_change"), ascending=False)
            
            # Display table with formatted columns
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    translate("popularity_change"): st.column_config.NumberColumn(
                        translate("popularity_change"),
                        format="%.2f",
                        help=translate("popularity_change"),
                    ),
                    "Change %": st.column_config.NumberColumn(
                        "Change %",
                        format="%.1f%%",
                        help="Percentage change in popularity",
                    ),
                },
            )
            
            # Visualize top changes
            top_n = min(TOP_TOPICS_DISPLAY_LIMIT, len(df))
            top_changes = df.head(top_n)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_changes[translate("title")],
                y=top_changes[translate("popularity_change")],
                marker_color=top_changes[translate("popularity_change")].apply(
                    lambda x: 'green' if x > 0 else 'red'
                ),
                text=top_changes[translate("popularity_change")].round(2),
                textposition='outside',
            ))
            
            fig.update_layout(
                title=translate("popularity_change") + f" (Top {top_n})",
                xaxis_title=translate("topic"),
                yaxis_title=translate("popularity_change"),
                height=DEFAULT_CHART_HEIGHT,
                xaxis={'tickangle': -45},
                hovermode='x',
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(translate("no_data"))
    
    except Exception as e:
        st.error(f"Error comparing stable topics: {e}", icon=ERROR_ICON)


def compare_source_diversity(
    data1: dict[str, pd.DataFrame], 
    data2: dict[str, pd.DataFrame], 
    period1: datetime, 
    period2: datetime
) -> None:
    """
    Compare average source diversity between two periods.
    
    Calculates and visualizes the average source diversity for weak and strong signals
    in both periods using a grouped bar chart.
    
    Args:
        data1: Dictionary of DataFrames for the first period
        data2: Dictionary of DataFrames for the second period
        period1: First period timestamp
        period2: Second period timestamp
    """
    st.subheader(translate("source_diversity_comparison"))
    
    try:
        diversity_data = []
        
        for category_name, category_key in [
            (translate("weak_signals"), WEAK_SIGNALS),
            (translate("strong_signals"), STRONG_SIGNALS),
        ]:
            df1 = data1.get(category_key, pd.DataFrame())
            df2 = data2.get(category_key, pd.DataFrame())
            
            avg_diversity1 = (
                df1[COLUMN_SOURCE_DIVERSITY].mean() 
                if not df1.empty and COLUMN_SOURCE_DIVERSITY in df1.columns 
                else 0
            )
            avg_diversity2 = (
                df2[COLUMN_SOURCE_DIVERSITY].mean() 
                if not df2.empty and COLUMN_SOURCE_DIVERSITY in df2.columns 
                else 0
            )
            
            diversity_data.append({
                "Category": category_name,
                period1.strftime(DATE_FORMAT): avg_diversity1,
                period2.strftime(DATE_FORMAT): avg_diversity2,
                "Change": avg_diversity2 - avg_diversity1,
            })
        
        df_diversity = pd.DataFrame(diversity_data)
        
        # Create grouped bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=period1.strftime(DATE_FORMAT),
            x=df_diversity["Category"],
            y=df_diversity[period1.strftime(DATE_FORMAT)],
            marker_color='lightgreen',
            text=df_diversity[period1.strftime(DATE_FORMAT)].round(2),
            textposition='auto',
        ))
        fig.add_trace(go.Bar(
            name=period2.strftime(DATE_FORMAT),
            x=df_diversity["Category"],
            y=df_diversity[period2.strftime(DATE_FORMAT)],
            marker_color='darkgreen',
            text=df_diversity[period2.strftime(DATE_FORMAT)].round(2),
            textposition='auto',
        ))
        
        fig.update_layout(
            barmode='group',
            xaxis_title=translate("signal_type"),
            yaxis_title=translate("source_diversity_comparison"),
            height=DEFAULT_CHART_HEIGHT,
            hovermode='x unified',
            showlegend=True,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display summary metrics
        col1, col2 = st.columns(2)
        for idx, row in df_diversity.iterrows():
            with col1 if idx == 0 else col2:
                st.metric(
                    label=row["Category"],
                    value=f"{row[period2.strftime(DATE_FORMAT)]:.2f}",
                    delta=f"{row['Change']:.2f}",
                )
    
    except Exception as e:
        st.error(f"Error comparing source diversity: {e}", icon=ERROR_ICON)


@st.fragment()
def dashboard_comparative() -> None:
    """
    Main dashboard for comparative analysis between different time periods.
    
    This function orchestrates the comparative analysis workflow:
    1. Period selection for comparison
    2. Data loading with validation
    3. Topic count comparison visualization
    4. Topic evolution analysis (new, disappeared, stable)
    5. Source diversity comparison
    
    The dashboard provides interactive visualizations and metrics to help users
    understand how topics and their characteristics have changed between two periods.
    """
    st.header(ANALYSIS_ICON + " " + translate("comparative_analysis_title"))
    
    try:
        # Select two periods to compare
        model_id, period_1, period_2 = choose_two_periods()
        
        # Validate that periods are different
        if period_1 == period_2:
            st.warning(translate("select_two_periods"), icon=WARNING_ICON)
            st.stop()
        
        # Load data for both periods with progress indication
        username = st.session_state.get("username", "")
        if not username:
            st.error("Username not found in session state", icon=ERROR_ICON)
            st.stop()
        
        with st.spinner(translate("loading_data") if hasattr(translate, "loading_data") else "Loading data..."):
            data_period_1 = get_period_data(model_id, period_1, username)
            data_period_2 = get_period_data(model_id, period_2, username)
        
        # Validate data availability
        period_1_has_data = any(not df.empty for df in data_period_1.values())
        period_2_has_data = any(not df.empty for df in data_period_2.values())
        
        if not period_1_has_data or not period_2_has_data:
            st.error(translate("no_data_for_comparison"), icon=WARNING_ICON)
            if not period_1_has_data:
                st.info(f"No data available for period: {period_1.strftime(DATE_FORMAT)}")
            if not period_2_has_data:
                st.info(f"No data available for period: {period_2.strftime(DATE_FORMAT)}")
            st.stop()
        
        # Display comparison sections
        st.divider()
        
        # Section 1: Topic count comparison
        with st.container():
            compare_topic_counts(data_period_1, data_period_2, period_1, period_2)
        
        st.divider()
        
        # Section 2: Topic evolution analysis
        with st.container():
            analyze_topic_evolution(data_period_1, data_period_2, period_1, period_2)
        
        st.divider()
        
        # Section 3: Source diversity comparison
        with st.container():
            compare_source_diversity(data_period_1, data_period_2, period_1, period_2)
    
    except Exception as e:
        st.error(f"Error in comparative dashboard: {e}", icon=ERROR_ICON)
        # Log the error for debugging (in production, use proper logging)
        import traceback
        with st.expander("Error details (for debugging)"):
            st.code(traceback.format_exc())
