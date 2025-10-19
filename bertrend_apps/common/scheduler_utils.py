#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import locale
from abc import abstractmethod, ABC
from pathlib import Path

from cron_descriptor import (
    Options,
    CasingTypeEnum,
    ExpressionDescriptor,
    DescriptionTypeEnum,
)

from bertrend import BEST_CUDA_DEVICE
from bertrend.demos.demos_utils.i18n import get_current_internationalization_language


class SchedulerUtils(ABC):
    @staticmethod
    def get_understandable_cron_description(cron_expression: str) -> str:
        """Returns a human understandable crontab description."""
        # Save current locale
        saved_locale = locale.setlocale(locale.LC_ALL)

        options = Options()
        options.casing_type = CasingTypeEnum.Sentence
        options.use_24hour_time_format = True

        locale_code = (
            "fr_FR.UTF-8"
            if get_current_internationalization_language() == "fr"
            else "en_US.UTF-8"
        )
        crontab_locale_code = (
            "fr_FR" if get_current_internationalization_language() == "fr" else "en"
        )

        options.locale_code = crontab_locale_code

        try:
            # Set temporary locale to specific locale
            locale.setlocale(locale.LC_ALL, locale_code)
            descriptor = ExpressionDescriptor(cron_expression, options)
            description = descriptor.get_description(DescriptionTypeEnum.FULL)

        finally:
            # Restore original locale
            locale.setlocale(locale.LC_ALL, saved_locale)

        return description

    @abstractmethod
    def add_job_to_crontab(self, schedule, command, env_vars="") -> bool:
        """Add the specified job to the crontab."""
        pass

    @abstractmethod
    def check_cron_job(self, pattern: str) -> bool:
        """Check if a specific pattern (expressed as a regular expression) matches crontab entries."""
        pass

    @abstractmethod
    def remove_from_crontab(self, pattern: str) -> bool:
        """Removes from the crontab the job matching the provided pattern (expressed as a regular expression)"""
        pass

    @abstractmethod
    def schedule_scrapping(self, feed_cfg: Path, user: str = None):
        """Schedule data scrapping on the basis of a feed configuration file"""
        pass

    @abstractmethod
    def schedule_newsletter(
        self,
        newsletter_cfg_path: Path,
        data_feed_cfg_path: Path,
        cuda_devices: str = BEST_CUDA_DEVICE,
    ):
        """Schedule data scrapping on the basis of a feed configuration file"""
        pass

    def check_if_scrapping_active_for_user(
        self, feed_id: str, user: str | None = None
    ) -> bool:
        """Checks if a given scrapping feed is active (registered with the service)."""
        if user:
            return self.check_cron_job(
                rf"scrape-feed.*/users/{user}/{feed_id}_feed.toml"
            )
        else:
            return self.check_cron_job(rf"scrape-feed.*/{feed_id}_feed.toml")

    def remove_scrapping_for_user(self, feed_id: str, user: str | None = None):
        """Removes from the scheduler service the job matching the provided feed_id"""
        if user:
            return self.remove_from_crontab(
                rf"scrape-feed.*/users/{user}/{feed_id}_feed.toml"
            )
        else:
            return self.remove_from_crontab(rf"scrape-feed.*/{feed_id}_feed.toml")
