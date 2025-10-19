#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
"""
Utilities to schedule jobs using the internal scheduling_service instead of system crontab.
This module mirrors the public API of crontab_utils.py so it can be used as a drop-in
replacement where we want to rely on the new scheduler service.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from bertrend import BEST_CUDA_DEVICE, BERTREND_LOG_PATH, load_toml_config
import requests
from urllib.parse import urljoin, quote

load_dotenv(override=True)

# Base URL for the scheduling service (FastAPI). Can be overridden via env var.
SCHEDULER_SERVICE_URL = os.getenv("SCHEDULER_SERVICE_URL", "http://localhost:8000/")

# Single shared session for connection pooling
_session = requests.Session()
_REQUEST_TIMEOUT = float(os.getenv("SCHEDULER_HTTP_TIMEOUT", "5"))


def _request(method: str, path: str, *, json: dict | None = None):
    url = urljoin(SCHEDULER_SERVICE_URL, path)
    resp = _session.request(method.upper(), url, json=json, timeout=_REQUEST_TIMEOUT)
    return resp


def _job_id_from_string(s: str) -> str:
    """Generate a deterministic, short job id from an arbitrary string."""
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    return f"job_{h}"


def _list_jobs() -> list[dict]:
    """Return list of jobs using the HTTP API."""
    r = _request("GET", "/jobs")
    if r.status_code != 200:
        logger.error(f"Failed to list jobs: {r.status_code} {r.text}")
        return []
    try:
        return r.json()
    except Exception:
        return []


def get_understandable_cron_description(cron_expression: str) -> str:
    """Returns a human understandable crontab description using scheduling_service HTTP API."""
    r = _request("POST", "/cron/validate", json={"expression": cron_expression})
    if r.status_code != 200:
        logger.error(f"Cron validation failed: {r.status_code} {r.text}")
        return cron_expression
    data = r.json()
    return data.get("description", cron_expression)


def add_job_to_crontab(schedule: str, command: str, env_vars: str = "") -> bool:
    """Add the specified job to the scheduler service via HTTP.

    We preserve the original signature; internally we create a cron job calling
    the service's sample_job with the command embedded as a message. To allow
    regex-based discovery like the legacy crontab, we embed the command text in
    the job_id so that listing jobs exposes it.
    """
    logger.debug(f"Scheduling via service (HTTP): {schedule} {env_vars} {command}")
    # Make job id readable for regex checks
    job_id = f"cron|{schedule}|{(env_vars + ' ').strip()}{command}"
    payload = {
        "job_id": job_id,
        "job_type": "cron",
        "function_name": "sample_job",
        "cron_expression": schedule,
        "args": [],
        "kwargs": {"message": f"{env_vars} {command}".strip()},
        "max_instances": 3,
        "coalesce": True,
    }
    r = _request("POST", "/jobs", json=payload)
    if r.status_code in (200, 201):
        return True
    # Consider duplicates as success
    try:
        detail = r.json().get("detail", "")
    except Exception:
        detail = r.text
    if "already exists" in str(detail):
        return True
    logger.error(f"Failed to create job: {r.status_code} {detail}")
    return False


def check_cron_job(pattern: str) -> bool:
    """Check if a specific regex pattern matches any scheduled service job.

    We search over job id, name and trigger string returned by the service. The
    job_id includes the original command text, so existing regex patterns keep working.
    """
    try:
        regex = re.compile(pattern)
    except re.error:
        logger.error(f"Invalid regex pattern: {pattern}")
        return False

    for job in _list_jobs():
        hay = " ".join(
            [
                str(job.get("job_id", "")),
                str(job.get("name", "")),
                str(job.get("trigger", "")),
            ]
        )
        if regex.search(hay):
            return True
    return False


def remove_from_crontab(pattern: str) -> bool:
    """Remove jobs from the scheduler service whose properties match the regex pattern via HTTP."""
    try:
        regex = re.compile(pattern)
    except re.error:
        logger.error(f"Invalid regex pattern: {pattern}")
        return False

    to_delete: list[str] = []
    for job in _list_jobs():
        hay = " ".join(
            [
                str(job.get("job_id", "")),
                str(job.get("name", "")),
                str(job.get("trigger", "")),
            ]
        )
        if regex.search(hay):
            to_delete.append(job.get("job_id"))

    if not to_delete:
        logger.warning("No job matching the provided pattern")
        return False

    ok = True
    for jid in to_delete:
        # URL-encode the id for path safety
        r = _request("DELETE", f"/jobs/{quote(jid, safe='')}")
        if r.status_code != 200:
            ok = False
            logger.error(f"Failed to delete job {jid}: {r.status_code} {r.text}")
    return ok


def schedule_scrapping(feed_cfg: Path, user: str | None = None):
    """Schedule data scrapping based on a feed configuration file using the service.

    We keep the same semantics for building the command string (for traceability in
    logs and for regex checks), but the scheduler will run the service's sample_job
    carrying this command as a message.
    """
    data_feed_cfg = load_toml_config(feed_cfg)
    schedule = data_feed_cfg["data-feed"]["update_frequency"]
    id = data_feed_cfg["data-feed"]["id"]

    # Prepare log path like the original util for consistency
    log_path = BERTREND_LOG_PATH if not user else BERTREND_LOG_PATH / "users" / user
    log_path.mkdir(parents=True, exist_ok=True)

    command = (
        f"{sys.executable} -m bertrend_apps.data_provider scrape-feed {feed_cfg.resolve()} > "
        f"{log_path}/cron_feed_{id}.log 2>&1"
    )

    # Use schedule+command in job_id to keep determinism and allow pattern search
    add_job_to_crontab(schedule, command, "")


def schedule_newsletter(
    newsletter_cfg_path: Path,
    data_feed_cfg_path: Path,
    cuda_devices: str = BEST_CUDA_DEVICE,
):
    """Schedule newsletter generation based on configuration using the service."""
    newsletter_cfg = load_toml_config(newsletter_cfg_path)
    schedule = newsletter_cfg["newsletter"]["update_frequency"]
    id = newsletter_cfg["newsletter"]["id"]
    command = (
        f"{sys.executable} -m bertrend_apps.newsletters newsletters "
        f"{newsletter_cfg_path.resolve()} {data_feed_cfg_path.resolve()} > "
        f"{BERTREND_LOG_PATH}/cron_newsletter_{id}.log 2>&1"
    )
    env_vars = f"CUDA_VISIBLE_DEVICES={cuda_devices}"
    add_job_to_crontab(schedule, command, env_vars)


def check_if_scrapping_active_for_user(feed_id: str, user: str | None = None) -> bool:
    """Checks if a given scrapping feed is active (registered with the service)."""
    if user:
        return check_cron_job(rf"scrape-feed.*/users/{user}/{feed_id}_feed.toml")
    else:
        return check_cron_job(rf"scrape-feed.*/{feed_id}_feed.toml")


def remove_scrapping_for_user(feed_id: str, user: str | None = None):
    """Removes from the scheduler service the job matching the provided feed_id"""
    if user:
        return remove_from_crontab(rf"scrape-feed.*/users/{user}/{feed_id}_feed.toml")
    else:
        return remove_from_crontab(rf"scrape-feed.*/{feed_id}_feed.toml")
