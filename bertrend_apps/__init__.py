#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import os

from dotenv import load_dotenv
from loguru import logger

from bertrend_apps.common.apscheduler_utils import APSchedulerUtils
from bertrend_apps.common.crontab_utils import CrontabSchedulerUtils


# Charger le fichier .env au démarrage du module
if load_dotenv(override=True):
    logger.info("Loaded .env file for scheduler configuration")
else:
    logger.warning("Failed to load .env file, using default scheduler configuration")


# Charger le type de scheduler depuis la variable d'environnement
SCHEDULER_TYPE = os.getenv("SCHEDULER_TYPE", "crontab").lower()

if SCHEDULER_TYPE == "apscheduler":
    SCHEDULER_UTILS = APSchedulerUtils()
else:  # assume default="crontab":
    SCHEDULER_UTILS = CrontabSchedulerUtils()
