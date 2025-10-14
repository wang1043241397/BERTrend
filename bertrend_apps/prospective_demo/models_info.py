#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import multiprocessing
import os
import random
import shutil
import sys
import time

import pandas as pd
import streamlit as st
import toml
from loguru import logger

from bertrend import BERTREND_LOG_PATH, BEST_CUDA_DEVICE, load_toml_config
from bertrend.config.parameters import LANGUAGES
from bertrend.demos.demos_utils.i18n import translate
from bertrend.demos.demos_utils.icons import (
    EDIT_ICON,
    DELETE_ICON,
    WARNING_ICON,
    INFO_ICON,
    TOGGLE_ON_ICON,
    TOGGLE_OFF_ICON,
    ERROR_ICON,
    RESTART_ICON,
)
from bertrend.demos.streamlit_components.clickable_df_component import clickable_df
from bertrend.demos.streamlit_components.input_with_pills_component import (
    input_with_pills,
)
from bertrend_apps.common.crontab_utils import (
    check_cron_job,
    remove_from_crontab,
    add_job_to_crontab,
)
from bertrend_apps.prospective_demo import (
    get_user_models_path,
    get_model_cfg_path,
    DEFAULT_ANALYSIS_CFG,
)
from bertrend_apps.prospective_demo.perf_utils import get_least_used_gpu
from bertrend_apps.prospective_demo.process_new_data import regenerate_models
from bertrend_apps.prospective_demo.utils import is_valid_email


@st.cache_data(ttl=60)
def load_model_config(model_id: str, username: str):
    """Loads the model config from the disk"""
    return load_toml_config(get_model_cfg_path(user_name=username, model_id=model_id))


@st.fragment
def models_monitoring():
    if not st.session_state.user_feeds:
        st.stop()

    st.session_state.model_analysis_cfg = {}
    displayed_list = []

    for model_id in sorted(st.session_state.user_feeds.keys()):
        try:
            st.session_state.model_analysis_cfg[model_id] = load_model_config(
                model_id, st.session_state.username
            )
        except Exception:
            # create default config if not found
            st.session_state.model_analysis_cfg[model_id] = DEFAULT_ANALYSIS_CFG
            # align language with the feed
            st.session_state.model_analysis_cfg[model_id]["model_config"][
                "language"
            ] = st.session_state.user_feeds[model_id]["data-feed"]["language"]
            # special case for split by paragraphs
            st.session_state.model_analysis_cfg[model_id]["model_config"][
                "split_by_paragraph"
            ] = (
                st.session_state.user_feeds[model_id]["data-feed"]["provider"]
                != "arxiv"
            )
            save_model_config(model_id, st.session_state.model_analysis_cfg[model_id])

        list_models = get_models_info(model_id, st.session_state.username)
        displayed_list.append(
            {
                translate("col_id"): model_id,
                translate("col_num_models"): len(list_models) if list_models else 0,
                translate("col_first_model_date"): (
                    list_models[0] if list_models else None
                ),
                translate("col_last_model_date"): (
                    list_models[-1] if list_models else None
                ),
                translate("col_update_frequency"): st.session_state.model_analysis_cfg[
                    model_id
                ]["model_config"]["granularity"],
                translate("col_analysis_window"): st.session_state.model_analysis_cfg[
                    model_id
                ]["model_config"]["window_size"],
            }
        )

    st.session_state.models_paths = {
        model_id: get_user_models_path(st.session_state.username, model_id)
        for model_id in st.session_state.user_feeds.keys()
    }

    df = pd.DataFrame(displayed_list)
    if not df.empty:
        df = df.sort_values(by=translate("col_id"), inplace=False).reset_index(
            drop=True
        )

    clickable_df_buttons = [
        (EDIT_ICON, edit_model_parameters, "secondary"),
        (lambda x: toggle_icon(df, x), handle_toggle_learning, "secondary"),
        (DELETE_ICON, handle_delete_models, "primary"),
        (RESTART_ICON, handle_regenerate_models, "primary"),
    ]
    clickable_df(df, clickable_df_buttons)


@st.dialog(translate("dialog_parameters"), width="large")
def edit_model_parameters(row_dict: dict):
    model_id = row_dict[translate("col_id")]
    st.write(f"**{translate('model_params_title').format(model_id)}**")
    c1, c2 = st.columns([1, 1])
    with c1:
        new_granularity = st.slider(
            translate("update_frequency_label"),
            min_value=1,
            max_value=30,
            value=st.session_state.model_analysis_cfg[model_id]["model_config"][
                "granularity"
            ],
            step=1,
            help=f"{INFO_ICON} {translate('update_frequency_help')}",
        )
    with c2:
        new_window_size = st.slider(
            translate("time_window_label"),
            min_value=new_granularity,
            max_value=30,
            value=max(
                st.session_state.model_analysis_cfg[model_id]["model_config"][
                    "window_size"
                ],
                new_granularity,
            ),
            step=1,
            help=f"{INFO_ICON} {translate('time_window_help')}",
        )

    split_by_paragraph = st.checkbox(
        translate("split_by_paragraph"),
        value=st.session_state.model_analysis_cfg[model_id]["model_config"][
            "split_by_paragraph"
        ],
        help=f"{INFO_ICON} {translate('split_by_paragraph_help')}",
    )

    language = st.segmented_control(
        translate("feed_language_label"),
        selection_mode="single",
        options=LANGUAGES,
        default=(
            LANGUAGES[0]
            if st.session_state.model_analysis_cfg[model_id]["model_config"]["language"]
            == "fr"
            else LANGUAGES[1]
        ),
        format_func=lambda lang: translate(f"language_{lang.lower()}"),
        help=translate("feed_language_help"),
    )

    st.write(f"**{translate('analysis_params_title').format(model_id)}**")
    cols = st.columns(3)
    with cols[0]:
        topic_evolution = st.checkbox(
            translate("topic_evolution"),
            value=st.session_state.model_analysis_cfg[model_id]["analysis_config"][
                "topic_evolution"
            ],
        )
    with cols[1]:
        evolution_scenarios = st.checkbox(
            translate("evolution_scenarios"),
            value=st.session_state.model_analysis_cfg[model_id]["analysis_config"][
                "evolution_scenarios"
            ],
        )
    with cols[2]:
        multifactorial_analysis = st.checkbox(
            translate("multifactorial_analysis"),
            value=st.session_state.model_analysis_cfg[model_id]["analysis_config"][
                "multifactorial_analysis"
            ],
        )

    st.write(f"**{translate('report_params_title').format(model_id)}**")

    # Get current report_config or use defaults
    current_report_config = st.session_state.model_analysis_cfg[model_id].get(
        "report_config", DEFAULT_ANALYSIS_CFG["report_config"]
    )

    auto_send = st.checkbox(
        translate("auto_send_reports"),
        value=current_report_config.get("auto_send", False),
        help=f"{INFO_ICON} {translate('auto_send_reports_help')}",
    )

    report_title = ""
    email_recipients = []
    max_emerging_topics = 1
    max_strong_topics = 1
    if auto_send:
        report_title = st.text_input(
            translate("report_title_label"),
            value=current_report_config.get("report_title", ""),
            help=f"{INFO_ICON} {translate('report_title_help')}",
        )
        email_recipients = input_with_pills(
            label=translate("email_recipients_label"),
            placeholder=translate("email_recipients"),
            validate_fn=is_valid_email,
            help=f"{INFO_ICON} {translate('email_recipients_help')}",
            value=current_report_config.get("email_recipients", []),
            key=f"{model_id}_config_email_recipients",
        )

        cols = st.columns(2)
        with cols[0]:
            max_emerging_topics = st.slider(
                translate("max_emerging_topics_label"),
                min_value=1,
                max_value=10,
                value=current_report_config.get("max_emerging_topics", 3),
                step=1,
                help=f"{INFO_ICON} {translate('max_emerging_topics_help')}",
            )
        with cols[1]:
            max_strong_topics = st.slider(
                translate("max_strong_topics_label"),
                min_value=1,
                max_value=10,
                value=current_report_config.get("max_strong_topics", 5),
                step=1,
                help=f"{INFO_ICON} {translate('max_strong_topics_help')}",
            )

    model_config = {
        "granularity": new_granularity,
        "window_size": new_window_size,
        "language": "fr" if language == "French" else "en",
        "split_by_paragraph": split_by_paragraph,
    }
    analysis_config = {
        "topic_evolution": topic_evolution,
        "evolution_scenarios": evolution_scenarios,
        "multifactorial_analysis": multifactorial_analysis,
    }
    report_config = {
        "auto_send": auto_send,
        "email_recipients": email_recipients,
        "report_title": report_title,
        "max_emerging_topics": max_emerging_topics,
        "max_strong_topics": max_strong_topics,
    }

    if st.button(translate("btn_ok")):
        save_model_config(
            model_id,
            {
                "model_config": model_config,
                "analysis_config": analysis_config,
                "report_config": report_config,
            },
        )
        # reload model config to update correctly memory cache
        st.cache_data.clear()
        st.session_state.model_analysis_cfg[model_id] = load_model_config(
            model_id, st.session_state.username
        )

        update_scheduled_training_for_user(model_id, st.session_state.username)
        update_scheduled_report_generation_for_user(model_id, st.session_state.username)
        st.rerun()


def save_model_config(model_id: str, config: dict):
    model_cfg_path = get_model_cfg_path(
        user_name=st.session_state.username, model_id=model_id
    )
    with open(model_cfg_path, "w") as toml_file:
        toml.dump(config, toml_file)
    logger.debug(f"Saved model analysis config {config} to {model_cfg_path}")


@st.dialog(translate("dialog_confirmation"))
def handle_delete_models(row_dict: dict):
    """Function to handle remove models from cache"""
    model_id = row_dict[translate("col_id")]
    st.warning(
        f":orange[{WARNING_ICON}] {translate('delete_models_warning').format(model_id)}"
    )
    col1, col2, _ = st.columns([2, 2, 8])
    with col1:
        if st.button(translate("btn_yes"), type="primary"):
            remove_scheduled_training_for_user(
                model_id=model_id, user=st.session_state.username
            )
            delete_cached_models(model_id)
            logger.info(translate("models_deleted_success").format(model_id))
            # Clear cache to reflect deleted models
            st.cache_data.clear()
            time.sleep(0.2)
            st.rerun()
    with col2:
        if st.button(translate("btn_no")):
            st.rerun()


@st.dialog(translate("dialog_model_regeneration"))
def handle_regenerate_models(row_dict: dict):
    """Function to regenerate models from scratch"""
    model_id = row_dict[translate("col_id")]
    st.warning(
        f"{WARNING_ICON} {translate('regenerate_models_warning').format(model_id)}"
    )
    st.warning(f"{WARNING_ICON} {translate('regenerate_models_delete_warning')}")
    st.error(f"{ERROR_ICON} {translate('regenerate_models_irreversible')}")
    col1, col2, _ = st.columns([2, 2, 8])
    with col1:
        if yes_btn := st.button(translate("btn_yes"), type="primary"):
            # Delete previously stored model
            delete_cached_models(model_id)
            logger.info(translate("models_deleted_success").format(model_id))
            # Clear cache to reflect deleted models
            st.cache_data.clear()

            # Regenerate new models
            # Launch model generation in a separate thread to avoid blocking the app
            # Create a new process with parameters
            # Set the start method to 'spawn' - required for a process using CUDA
            os.environ["CUDA_VISIBLE_DEVICES"] = get_least_used_gpu()
            multiprocessing.set_start_method("spawn", force=True)

            process = multiprocessing.Process(
                target=regenerate_models,
                kwargs={"model_id": model_id, "user": st.session_state.username},
            )
            # Start the process
            process.start()

            time.sleep(0.2)
            # st.rerun()

    with col2:
        if st.button(translate("btn_no")):
            st.rerun()

    if yes_btn:
        st.info(f"{INFO_ICON} {translate('regeneration_in_progress').format(model_id)}")
        st.info(f"{INFO_ICON} {translate('regeneration_close_info')}")


def toggle_learning(cfg: dict):
    """Activate / deactivate the learning from the crontab"""
    model_id = cfg[translate("col_id")]
    if check_if_learning_active_for_user(
        model_id=model_id, user=st.session_state.username
    ):
        if remove_scheduled_training_for_user(
            model_id=model_id, user=st.session_state.username
        ):
            st.toast(
                translate("learning_deactivated").format(model_id),
                icon=INFO_ICON,
            )
            logger.info(f"Learning for {model_id} deactivated !")
            # Also remove report generation if it was scheduled
            if check_if_report_generation_active_for_user(
                model_id, st.session_state.username
            ):
                remove_scheduled_report_generation_for_user(
                    model_id, st.session_state.username
                )
                logger.info(f"Automated report generation for {model_id} deactivated !")
    else:
        schedule_training_for_user(model_id, st.session_state.username)
        st.toast(translate("learning_activated").format(model_id), icon=WARNING_ICON)
        logger.info(f"Learning for {model_id} activated !")
        # Also schedule report generation if configured
        if schedule_report_generation_for_user(model_id, st.session_state.username):
            logger.info(f"Automated report generation for {model_id} activated !")
    # Clear cache to reflect updated crontab state
    st.cache_data.clear()
    time.sleep(0.2)
    st.rerun()


@st.dialog(translate("dialog_confirmation"))
def handle_toggle_learning(cfg: dict):
    """Function to handle remove click events"""
    model_id = cfg[translate("col_id")]
    if check_if_learning_active_for_user(
        model_id=model_id, user=st.session_state.username
    ):
        st.write(
            f":orange[{WARNING_ICON}] {translate('deactivate_learning_question').format(model_id)}"
        )
        col1, col2, _ = st.columns([2, 2, 8])
        with col1:
            if st.button(translate("btn_yes"), type="primary"):
                toggle_learning(cfg)
                st.rerun()
        with col2:
            if st.button(translate("btn_no")):
                st.rerun()
    else:
        st.write(
            f":blue[{INFO_ICON}] {translate('activate_learning_info').format(model_id)}"
        )
        toggle_learning(cfg)
        st.rerun()


def toggle_icon(df: pd.DataFrame, index: int) -> str:
    """Switch the toggle icon depending on the status of the scrapping feed in the crontab"""
    model_id = df[translate("col_id")][index]
    return (
        f":green[{TOGGLE_ON_ICON}]"
        if check_if_learning_active_for_user(
            model_id=model_id, user=st.session_state.username
        )
        else f":red[{TOGGLE_OFF_ICON}]"
    )


@st.cache_data(ttl=30)
def check_if_learning_active_for_user(model_id: str, user: str):
    """Checks if a given scrapping feed is active (registered in the crontab"""
    if user:
        return check_cron_job(rf"process_new_data train-new-model.*{user}.*{model_id}")
    else:
        return False


def remove_scheduled_training_for_user(model_id: str, user: str):
    """Removes from the crontab the training job matching the provided model_id"""
    if user:
        return remove_from_crontab(
            rf"process_new_data train-new-model {user} {model_id}"
        )
    return False


def update_scheduled_training_for_user(model_id: str, user: str):
    """Updates the crontab with the new training job"""
    if check_if_learning_active_for_user(model_id, user):
        remove_scheduled_training_for_user(model_id, user)
        schedule_training_for_user(model_id, user)
        return True
    return False


def schedule_training_for_user(model_id: str, user: str):
    """Schedule data scrapping on the basis of a feed configuration file"""
    schedule = generate_crontab_expression(
        st.session_state.model_analysis_cfg[model_id]["model_config"]["granularity"]
    )
    logpath = BERTREND_LOG_PATH / "users" / user
    logpath.mkdir(parents=True, exist_ok=True)
    command = (
        f"{sys.prefix}/bin/python -m bertrend_apps.prospective_demo.process_new_data train-new-model {user} {model_id} "
        f"> {logpath}/learning_{model_id}.log 2>&1"
    )
    env_vars = f"CUDA_VISIBLE_DEVICES={BEST_CUDA_DEVICE}"
    add_job_to_crontab(schedule, command, env_vars)


def delete_cached_models(model_id: str):
    """Removes models from the cache"""
    # Remove the directory and all its contents
    shutil.rmtree(st.session_state.models_paths[model_id])


def generate_crontab_expression(days_interval: int) -> str:
    # Random hour between 0 and 6 (inclusive)
    hour = random.randint(0, 6)  # run during the night
    # Random minute rounded to the nearest 10
    minute = random.choice([0, 10, 20, 30, 40, 50])
    # Compute days
    days = [str(i) for i in range(1, 31, days_interval)]
    # Crontab expression format: minute hour day_of_month month day_of_week
    crontab_expression = f"{minute} {hour} {','.join(days)} * *"
    return crontab_expression


def check_if_report_generation_active_for_user(model_id: str, user: str) -> bool:
    """Checks if automated report generation is active (registered in the crontab)"""
    if user:
        return check_cron_job(rf"automated_report_generation.*{user}.*{model_id}")
    else:
        return False


def remove_scheduled_report_generation_for_user(model_id: str, user: str) -> bool:
    """Removes from the crontab the report generation job matching the provided model_id"""
    if user:
        return remove_from_crontab(rf"automated_report_generation {user} {model_id}")
    return False


def schedule_report_generation_for_user(model_id: str, user: str) -> bool:
    """Schedule automated report generation based on model configuration"""
    # Check if report generation is configured in the model config
    if model_id not in st.session_state.model_analysis_cfg:
        logger.warning(f"Model config not found for {model_id}")
        return False

    report_config = st.session_state.model_analysis_cfg[model_id].get(
        "report_config", {}
    )
    auto_send = report_config.get("auto_send", False)
    recipients = report_config.get("email_recipients", [])

    if not auto_send:
        logger.info(f"auto_send is disabled for model {model_id}")
        return False

    if not recipients:
        logger.warning(f"No email recipients configured for model {model_id}")
        return False

    # Generate cron schedule based on granularity (same as training)
    schedule = generate_crontab_expression(
        st.session_state.model_analysis_cfg[model_id]["model_config"]["granularity"]
    )

    # Add a small delay after training completes (run 1 hour after training schedule)
    # Parse the schedule and add 1 hour
    parts = schedule.split()
    hour = int(parts[1])
    # Add 1 hour, wrapping around if necessary
    hour = (hour + 1) % 24
    parts[1] = str(hour)
    adjusted_schedule = " ".join(parts)

    logpath = BERTREND_LOG_PATH / "users" / user
    logpath.mkdir(parents=True, exist_ok=True)

    command = (
        f"{sys.prefix}/bin/python -m bertrend_apps.prospective_demo.automated_report_generation {user} {model_id} "
        f"> {logpath}/report_{model_id}.log 2>&1"
    )

    return add_job_to_crontab(adjusted_schedule, command, "")


def update_scheduled_report_generation_for_user(model_id: str, user: str) -> bool:
    """Updates the crontab with the new report generation job"""
    if check_if_report_generation_active_for_user(model_id, user):
        remove_scheduled_report_generation_for_user(model_id, user)
        return schedule_report_generation_for_user(model_id, user)
    return False


def safe_timestamp(x: str) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(x)
    except Exception as e:
        return None


@st.cache_data(ttl=60)
def get_models_info(model_id: str, username: str) -> list:
    """Returns the list of topic models that are stored, identified by their timestamp"""
    user_model_dir = get_user_models_path(username, model_id)
    if not user_model_dir.exists():
        return []
    matching_files = user_model_dir.glob(r"????-??-??")
    return sorted(
        [
            safe_timestamp(x.name)
            for x in matching_files
            if safe_timestamp(x.name) is not None
        ]
    )
