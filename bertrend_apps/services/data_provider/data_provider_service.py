#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from datetime import datetime, timedelta
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from bertrend import FEED_BASE_PATH, load_toml_config
from bertrend.article_scoring.article_scoring import QualityLevel
from bertrend_apps.common.crontab_utils import CrontabSchedulerUtils
from bertrend_apps.data_provider.arxiv_provider import ArxivProvider
from bertrend_apps.data_provider.atom_feed_provider import ATOMFeedProvider
from bertrend_apps.data_provider.rss_feed_provider import RSSFeedProvider
from bertrend_apps.data_provider.google_news_provider import GoogleNewsProvider
from bertrend_apps.data_provider.bing_news_provider import BingNewsProvider
from bertrend_apps.data_provider.newscatcher_provider import NewsCatcherProvider


# FastAPI application
app = FastAPI(title="Data Provider Service", version="1.0.0")

# Providers mapping (same as CLI)
PROVIDERS = {
    "arxiv": ArxivProvider,
    "atom": ATOMFeedProvider,
    "rss": RSSFeedProvider,
    "google": GoogleNewsProvider,
    "bing": BingNewsProvider,
    "newscatcher": NewsCatcherProvider,
}

scheduler_utils = CrontabSchedulerUtils()


# Request/Response models
class ScrapeRequest(BaseModel):
    keywords: str = Field(..., description="Keywords for data search engine.")
    provider: str = Field(
        default="google",
        description="source for data [arxiv, atom, rss, google, bing, newscatcher]",
    )
    after: Optional[str] = Field(
        default=None, description="date after which to consider news [YYYY-MM-DD]"
    )
    before: Optional[str] = Field(
        default=None, description="date before which to consider news [YYYY-MM-DD]"
    )
    max_results: int = Field(default=50, description="maximum results per request")
    save_path: Optional[Path] = Field(
        default=None, description="Path for writing results (jsonl)"
    )
    language: Optional[str] = Field(default=None, description="Language filter")


class ScrapeResponse(BaseModel):
    stored_path: Optional[Path]
    article_count: int


class AutoScrapeRequest(BaseModel):
    requests_file: Path = Field(
        ..., description="Path of input file containing the expected queries."
    )
    max_results: int = Field(default=50)
    provider: str = Field(
        default="google",
        description="source for news [arxiv, atom, rss, google, bing, newscatcher]",
    )
    save_path: Optional[Path] = None
    language: Optional[str] = None
    evaluate_articles_quality: bool = False
    minimum_quality_level: str = Field(default="AVERAGE")


class GenerateQueryFileRequest(BaseModel):
    keywords: str
    after: str
    before: str
    save_path: Path
    interval: int = Field(default=30, description="Range of days of atomic requests")


class GenerateQueryFileResponse(BaseModel):
    save_path: Path
    line_count: int


class ScrapeFeedRequest(BaseModel):
    feed_cfg: Path


class ScheduleScrappingRequest(BaseModel):
    feed_cfg: Path


# Utilities copied from CLI implementation


def _daterange(start_date: datetime, end_date: datetime, ndays: int):
    for n in range(int((end_date - start_date).days / ndays)):
        yield (
            start_date + timedelta(ndays * n),
            start_date + timedelta(ndays * (n + 1)),
        )


# Endpoints
@app.post("/scrape", response_model=ScrapeResponse)
def scrape(req: ScrapeRequest):
    provider_class = PROVIDERS.get(req.provider)
    if provider_class is None:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {req.provider}")
    provider = provider_class()
    results = provider.get_articles(
        req.keywords, req.after, req.before, req.max_results, req.language
    )
    provider.store_articles(results, req.save_path)
    return ScrapeResponse(stored_path=req.save_path, article_count=len(results))


@app.post("/auto-scrape", response_model=ScrapeResponse)
def auto_scrape(req: AutoScrapeRequest):
    provider_class = PROVIDERS.get(req.provider)
    if provider_class is None:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {req.provider}")
    provider = provider_class()
    try:
        with open(req.requests_file) as file:
            requests: List[List[str]] = [line.rstrip().split(";") for line in file]
    except Exception:
        raise HTTPException(status_code=400, detail="Bad file format")

    results = provider.get_articles_batch(
        queries_batch=requests,
        max_results=req.max_results,
        language=req.language,
        evaluate_articles_quality=req.evaluate_articles_quality,
        minimum_quality_level=QualityLevel.from_string(req.minimum_quality_level),
    )
    logger.info(f"Storing {len(results)} articles")
    provider.store_articles(results, req.save_path)
    return ScrapeResponse(stored_path=req.save_path, article_count=len(results))


@app.post("/generate-query-file", response_model=GenerateQueryFileResponse)
def generate_query_file(req: GenerateQueryFileRequest):
    date_format = "%Y-%m-%d"
    start_date = datetime.strptime(req.after, date_format)
    end_date = datetime.strptime(req.before, date_format)
    dates_l = list(_daterange(start_date, end_date, req.interval))

    line_count = 0
    with open(req.save_path, "a") as query_file:
        for elem in dates_l:
            query_file.write(
                f"{req.keywords};{elem[0].strftime(date_format)};{elem[1].strftime(date_format)}\n"
            )
            line_count += 1
    return GenerateQueryFileResponse(save_path=req.save_path, line_count=line_count)


@app.post("/scrape-feed", response_model=ScrapeResponse)
def scrape_from_feed(req: ScrapeFeedRequest):
    data_feed_cfg = load_toml_config(req.feed_cfg)
    current_date = datetime.today()
    current_date_str = current_date.strftime("%Y-%m-%d")
    days_to_subtract = data_feed_cfg["data-feed"].get("number_of_days")
    provider_name = data_feed_cfg["data-feed"].get("provider")
    keywords = data_feed_cfg["data-feed"].get("query")
    max_results = data_feed_cfg["data-feed"].get("max_results")
    before = current_date_str
    after = (current_date - timedelta(days=days_to_subtract)).strftime("%Y-%m-%d")
    language = data_feed_cfg["data-feed"].get("language")
    save_path = (
        FEED_BASE_PATH
        / data_feed_cfg["data-feed"].get("feed_dir_path")
        / f"{current_date_str}_{data_feed_cfg['data-feed'].get('id')}.jsonl"
    )
    evaluate_articles_quality = data_feed_cfg["data-feed"].get(
        "evaluate_articles_quality", False
    )
    minimum_quality_level = data_feed_cfg["data-feed"].get(
        "minimum_quality_level", QualityLevel.AVERAGE
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    article_count = 0
    with tempfile.NamedTemporaryFile() as query_file:
        if provider_name in {"arxiv", "atom", "rss"}:  # already returns batches
            resp = scrape(
                ScrapeRequest(
                    keywords=keywords,
                    provider=provider_name,
                    after=after,
                    before=before,
                    max_results=max_results,
                    save_path=save_path,
                    language=language,
                )
            )
            article_count = resp.article_count
        else:
            _ = generate_query_file(
                GenerateQueryFileRequest(
                    keywords=keywords,
                    after=after,
                    before=before,
                    interval=1,
                    save_path=Path(query_file.name),
                )
            )
            resp = auto_scrape(
                AutoScrapeRequest(
                    requests_file=Path(query_file.name),
                    max_results=max_results,
                    provider=provider_name,
                    save_path=save_path,
                    language=language,
                    evaluate_articles_quality=evaluate_articles_quality,
                    minimum_quality_level=str(minimum_quality_level),
                )
            )
            article_count = resp.article_count

    return ScrapeResponse(stored_path=save_path, article_count=article_count)


@app.post("/schedule-scrapping")
def automate_scrapping(req: ScheduleScrappingRequest):
    try:
        scheduler_utils.schedule_scrapping(req.feed_cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "scheduled"}
