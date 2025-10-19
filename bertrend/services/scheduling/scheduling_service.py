#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ProcessPoolExecutor
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Job Scheduler API", version="1.0.0")

# Configure job stores and executors
jobstores = {"default": SQLAlchemyJobStore(url="sqlite:///data/jobs.sqlite")}

executors = {"default": ProcessPoolExecutor(max_workers=5)}

job_defaults = {
    "coalesce": False,  # Run all missed executions
    "max_instances": 3,  # Maximum instances of the job running concurrently
}

# Initialize APScheduler with persistence (Paris timezone)
scheduler = BackgroundScheduler(
    jobstores=jobstores,
    executors=executors,
    job_defaults=job_defaults,
    timezone="Europe/Paris",
)
scheduler.start()


# Pydantic models
class JobCreate(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the job")
    job_name: Optional[str] = Field(None, description="Human-readable job name (defaults to job_id)")
    job_type: str = Field(..., description="Type: 'interval', 'cron', or 'date'")
    function_name: str = Field(..., description="Name of the function to execute")
    args: Optional[List[Any]] = Field(
        default=[], description="Arguments for the function"
    )
    kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="Keyword arguments"
    )

    # Interval trigger fields
    seconds: Optional[int] = None
    minutes: Optional[int] = None
    hours: Optional[int] = None
    days: Optional[int] = None

    # Cron trigger fields (two ways to specify)
    cron_expression: Optional[str] = Field(
        None,
        description="Cron expression (e.g., '0 12 * * *' for daily at noon) or use named fields below",
    )
    cron_minute: Optional[str] = Field(None, description="Minute (0-59 or */5 or 0,30)")
    cron_hour: Optional[str] = Field(None, description="Hour (0-23)")
    cron_day: Optional[str] = Field(None, description="Day of month (1-31)")
    cron_month: Optional[str] = Field(None, description="Month (1-12)")
    cron_day_of_week: Optional[str] = Field(
        None, description="Day of week (0-6, 0=Monday)"
    )

    # Date trigger fields
    run_date: Optional[datetime] = Field(
        None, description="Specific datetime to run once"
    )

    # Execution options
    max_instances: Optional[int] = Field(
        default=3, description="Max concurrent instances"
    )
    coalesce: Optional[bool] = Field(
        default=False, description="Coalesce missed executions"
    )


class JobUpdate(BaseModel):
    job_type: Optional[str] = None
    function_name: Optional[str] = None
    args: Optional[List[Any]] = None
    kwargs: Optional[Dict[str, Any]] = None
    seconds: Optional[int] = None
    minutes: Optional[int] = None
    hours: Optional[int] = None
    days: Optional[int] = None
    cron_expression: Optional[str] = None
    cron_minute: Optional[str] = None
    cron_hour: Optional[str] = None
    cron_day: Optional[str] = None
    cron_month: Optional[str] = None
    cron_day_of_week: Optional[str] = None
    run_date: Optional[datetime] = None
    max_instances: Optional[int] = None
    coalesce: Optional[bool] = None


class JobResponse(BaseModel):
    job_id: str
    name: str
    next_run_time: Optional[datetime]
    trigger: str
    executor: str
    max_instances: int


class JobExecutionResponse(BaseModel):
    message: str
    timestamp: datetime
    job_id: str


class CronExpressionRequest(BaseModel):
    expression: Optional[str] = Field(None, description="Standard cron expression")
    minute: Optional[str] = Field(None, description="Minute field")
    hour: Optional[str] = Field(None, description="Hour field")
    day: Optional[str] = Field(None, description="Day field")
    month: Optional[str] = Field(None, description="Month field")
    day_of_week: Optional[str] = Field(None, description="Day of week field")


class CronExpressionResponse(BaseModel):
    expression: str
    description: str
    next_runs: List[datetime]
    is_valid: bool
    timezone: str


# ====================================
# Job Functions (must be module-level for ProcessPoolExecutor)
# ====================================


def sample_job(message: str = "Default message"):
    """Example job function - must be at module level for multiprocessing"""
    import time

    logger.info(f"Executing job: {message}")
    print(f"[{datetime.now()}] Job executed: {message}")
    time.sleep(1)  # Simulate some work
    return f"Completed: {message}"


def cleanup_task():
    """Example cleanup task"""
    import time

    logger.info("Running cleanup task")
    print(f"[{datetime.now()}] Cleanup task executed")
    time.sleep(2)
    return "Cleanup completed"


def report_generator(report_type: str = "daily"):
    """Example report generator"""
    import time

    logger.info(f"Generating {report_type} report")
    print(f"[{datetime.now()}] Generated {report_type} report")
    time.sleep(1.5)
    return f"{report_type} report generated"


def data_processor(data_id: int, operation: str = "process"):
    """Example data processor"""
    import time

    logger.info(f"Processing data {data_id} with operation: {operation}")
    print(f"[{datetime.now()}] Processing data ID {data_id}: {operation}")
    time.sleep(1)
    return f"Processed data {data_id}"


def email_sender(recipient: str, subject: str = "Notification"):
    """Example email sender (simulated)"""
    import time

    logger.info(f"Sending email to {recipient}: {subject}")
    print(f"[{datetime.now()}] Email sent to {recipient}: {subject}")
    time.sleep(0.5)
    return f"Email sent to {recipient}"


# Job function registry
JOB_FUNCTIONS = {
    "sample_job": sample_job,
    "cleanup_task": cleanup_task,
    "report_generator": report_generator,
    "data_processor": data_processor,
    "email_sender": email_sender,
}


def get_trigger(job_data: JobCreate):
    """Create appropriate trigger based on job type"""
    if job_data.job_type == "interval":
        return IntervalTrigger(
            seconds=job_data.seconds or 0,
            minutes=job_data.minutes or 0,
            hours=job_data.hours or 0,
            days=job_data.days or 0,
        )
    elif job_data.job_type == "cron":
        # Support both cron_expression string and individual fields
        if job_data.cron_expression:
            parts = job_data.cron_expression.split()
            if len(parts) != 5:
                raise ValueError(
                    "Cron expression must have 5 parts: minute hour day month day_of_week"
                )
            return CronTrigger(
                minute=parts[0],
                hour=parts[1],
                day=parts[2],
                month=parts[3],
                day_of_week=parts[4],
                timezone="Europe/Paris",
            )
        elif any(
            [
                job_data.cron_minute,
                job_data.cron_hour,
                job_data.cron_day,
                job_data.cron_month,
                job_data.cron_day_of_week,
            ]
        ):
            return CronTrigger(
                minute=job_data.cron_minute or "*",
                hour=job_data.cron_hour or "*",
                day=job_data.cron_day or "*",
                month=job_data.cron_month or "*",
                day_of_week=job_data.cron_day_of_week or "*",
                timezone="Europe/Paris",
            )
        else:
            raise ValueError(
                "Either cron_expression or individual cron fields must be provided"
            )
    elif job_data.job_type == "date":
        if not job_data.run_date:
            raise ValueError("run_date is required for date jobs")
        return DateTrigger(run_date=job_data.run_date, timezone="Europe/Paris")
    else:
        raise ValueError(f"Invalid job_type: {job_data.job_type}")


@app.on_event("startup")
def startup_event():
    """Log startup information"""
    logger.info("FastAPI Job Scheduler started")
    logger.info(f"Job store: SQLite (jobs.sqlite)")
    logger.info(f"Executor: ProcessPoolExecutor (max_workers=5)")
    logger.info(f"Timezone: Europe/Paris")

    # Print existing jobs
    existing_jobs = scheduler.get_jobs()
    if existing_jobs:
        logger.info(f"Loaded {len(existing_jobs)} existing jobs from database")
        for job in existing_jobs:
            logger.info(f"  - {job.id}: next run at {job.next_run_time}")


@app.on_event("shutdown")
def shutdown_event():
    """Shutdown scheduler on app shutdown"""
    logger.info("Shutting down scheduler...")
    scheduler.shutdown(wait=True)
    logger.info("Scheduler shutdown complete")


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Job Scheduler API with Persistent Storage",
        "storage": "SQLite (jobs.sqlite)",
        "executor": "ProcessPoolExecutor (max_workers=5)",
        "timezone": "Europe/Paris",
        "endpoints": {
            "POST /jobs": "Create a new job",
            "GET /jobs": "List all jobs",
            "GET /jobs/{job_id}": "Get job details",
            "PUT /jobs/{job_id}": "Update a job",
            "DELETE /jobs/{job_id}": "Remove a job",
            "POST /jobs/{job_id}/pause": "Pause a job",
            "POST /jobs/{job_id}/resume": "Resume a job",
            "POST /jobs/{job_id}/run": "Run a job immediately",
            "GET /functions": "List available job functions",
            "POST /cron/validate": "Validate and preview cron expression",
            "GET /cron/examples": "Get cron expression examples",
        },
    }


@app.get("/functions")
def list_functions():
    """List available job functions"""
    return {
        "available_functions": list(JOB_FUNCTIONS.keys()),
        "details": {
            name: {
                "description": (
                    func.__doc__.strip() if func.__doc__ else "No description"
                ),
                "signature": str(
                    func.__code__.co_varnames[: func.__code__.co_argcount]
                ),
            }
            for name, func in JOB_FUNCTIONS.items()
        },
    }


@app.post("/jobs", response_model=JobResponse, status_code=201)
def create_job(job: JobCreate):
    """Create a new scheduled job"""
    try:
        # Check if job_id already exists
        if scheduler.get_job(job.job_id):
            raise HTTPException(
                status_code=400, detail=f"Job with id '{job.job_id}' already exists"
            )

        # Get the function to execute
        if job.function_name not in JOB_FUNCTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Function '{job.function_name}' not found. Available: {list(JOB_FUNCTIONS.keys())}",
            )

        func = JOB_FUNCTIONS[job.function_name]
        trigger = get_trigger(job)

        # Add job to scheduler with persistence
        scheduler.add_job(
            func,
            trigger=trigger,
            id=job.job_id,
            name=job.job_name or job.job_id,
            args=job.args,
            kwargs=job.kwargs,
            max_instances=job.max_instances,
            coalesce=job.coalesce,
            replace_existing=False,
        )

        added_job = scheduler.get_job(job.job_id)

        logger.info(f"Job '{job.job_id}' created successfully")

        return JobResponse(
            job_id=added_job.id,
            name=added_job.name,
            next_run_time=added_job.next_run_time,
            trigger=str(added_job.trigger),
            executor=added_job.executor,
            max_instances=added_job.max_instances,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating job: {str(e)}")


@app.get("/jobs", response_model=List[JobResponse])
def list_jobs():
    """List all scheduled jobs"""
    jobs = scheduler.get_jobs()
    return [
        JobResponse(
            job_id=job.id,
            name=job.name,
            next_run_time=job.next_run_time,
            trigger=str(job.trigger),
            executor=job.executor,
            max_instances=job.max_instances,
        )
        for job in jobs
    ]


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    """Get details of a specific job"""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return JobResponse(
        job_id=job.id,
        name=job.name,
        next_run_time=job.next_run_time,
        trigger=str(job.trigger),
        executor=job.executor,
        max_instances=job.max_instances,
    )


@app.put("/jobs/{job_id}", response_model=JobResponse)
def update_job(job_id: str, job_update: JobUpdate):
    """Update an existing job"""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    try:
        # Prepare update data
        update_data = job_update.dict(exclude_unset=True)

        # Update trigger if job_type or trigger parameters provided
        if any(
            k in update_data
            for k in [
                "job_type",
                "seconds",
                "minutes",
                "hours",
                "days",
                "cron_expression",
                "run_date",
            ]
        ):
            # Create a temporary JobCreate object for trigger generation
            job_type = update_data.get("job_type", "interval")
            temp_job = JobCreate(
                job_id=job_id,
                job_type=job_type,
                function_name="sample_job",
                seconds=update_data.get("seconds"),
                minutes=update_data.get("minutes"),
                hours=update_data.get("hours"),
                days=update_data.get("days"),
                cron_expression=update_data.get("cron_expression"),
                run_date=update_data.get("run_date"),
            )
            trigger = get_trigger(temp_job)
            scheduler.reschedule_job(job_id, trigger=trigger)

        # Update function if provided
        if "function_name" in update_data:
            if update_data["function_name"] not in JOB_FUNCTIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Function '{update_data['function_name']}' not found",
                )
            scheduler.modify_job(
                job_id, func=JOB_FUNCTIONS[update_data["function_name"]]
            )

        # Update args/kwargs if provided
        if "args" in update_data:
            scheduler.modify_job(job_id, args=update_data["args"])
        if "kwargs" in update_data:
            scheduler.modify_job(job_id, kwargs=update_data["kwargs"])

        # Update execution options
        if "max_instances" in update_data:
            scheduler.modify_job(job_id, max_instances=update_data["max_instances"])
        if "coalesce" in update_data:
            scheduler.modify_job(job_id, coalesce=update_data["coalesce"])

        updated_job = scheduler.get_job(job_id)

        logger.info(f"Job '{job_id}' updated successfully")

        return JobResponse(
            job_id=updated_job.id,
            name=updated_job.name,
            next_run_time=updated_job.next_run_time,
            trigger=str(updated_job.trigger),
            executor=updated_job.executor,
            max_instances=updated_job.max_instances,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating job: {str(e)}")


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    """Remove a scheduled job"""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    scheduler.remove_job(job_id)
    logger.info(f"Job '{job_id}' removed successfully")
    return {"message": f"Job '{job_id}' removed successfully"}


@app.post("/jobs/{job_id}/pause")
def pause_job(job_id: str):
    """Pause a job"""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    scheduler.pause_job(job_id)
    logger.info(f"Job '{job_id}' paused")
    return {"message": f"Job '{job_id}' paused"}


@app.post("/jobs/{job_id}/resume")
def resume_job(job_id: str):
    """Resume a paused job"""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    scheduler.resume_job(job_id)
    logger.info(f"Job '{job_id}' resumed")
    return {"message": f"Job '{job_id}' resumed"}


@app.post("/jobs/{job_id}/run", response_model=JobExecutionResponse)
def run_job_now(job_id: str):
    """Execute a job immediately (outside of its schedule)"""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    try:
        # Run the job immediately
        job.func(*job.args, **job.kwargs)
        logger.info(f"Job '{job_id}' executed manually")

        return JobExecutionResponse(
            message=f"Job '{job_id}' executed successfully",
            timestamp=datetime.now(),
            job_id=job_id,
        )
    except Exception as e:
        logger.error(f"Error executing job '{job_id}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing job: {str(e)}")


@app.post("/cron/validate", response_model=CronExpressionResponse)
def validate_cron(cron_req: CronExpressionRequest):
    """Validate a cron expression and show next execution times"""
    try:
        # Build cron expression
        if cron_req.expression:
            parts = cron_req.expression.split()
            if len(parts) != 5:
                raise ValueError(
                    "Cron expression must have 5 parts: minute hour day month day_of_week"
                )
            minute, hour, day, month, day_of_week = parts
            expression = cron_req.expression
        else:
            minute = cron_req.minute or "*"
            hour = cron_req.hour or "*"
            day = cron_req.day or "*"
            month = cron_req.month or "*"
            day_of_week = cron_req.day_of_week or "*"
            expression = f"{minute} {hour} {day} {month} {day_of_week}"

        # Create trigger and get next run times
        trigger = CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
            timezone="Europe/Paris",
        )

        # Get next 5 execution times
        now = datetime.now()
        next_runs = []
        current = now
        for _ in range(5):
            next_run = trigger.get_next_fire_time(None, current)
            if next_run:
                next_runs.append(next_run)
                current = next_run
            else:
                break

        # Generate human-readable description
        description = _describe_cron(minute, hour, day, month, day_of_week)

        return CronExpressionResponse(
            expression=expression,
            description=description,
            next_runs=next_runs,
            is_valid=True,
            timezone="Europe/Paris",
        )

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid cron expression: {str(e)}"
        )


def _describe_cron(
    minute: str, hour: str, day: str, month: str, day_of_week: str
) -> str:
    """Generate human-readable description of cron expression"""
    parts = []

    # Minute
    if minute == "*":
        parts.append("every minute")
    elif "/" in minute:
        interval = minute.split("/")[1]
        parts.append(f"every {interval} minutes")
    else:
        parts.append(f"at minute {minute}")

    # Hour
    if hour != "*":
        if "/" in hour:
            interval = hour.split("/")[1]
            parts.append(f"every {interval} hours")
        else:
            parts.append(f"at {hour}:00")

    # Day
    if day != "*":
        parts.append(f"on day {day}")

    # Month
    if month != "*":
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        if month.isdigit():
            parts.append(f"in {months[int(month)-1]}")
        else:
            parts.append(f"in month {month}")

    # Day of week
    if day_of_week != "*":
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if day_of_week.isdigit():
            parts.append(f"on {days[int(day_of_week)]}")
        else:
            parts.append(f"on day-of-week {day_of_week}")

    return ", ".join(parts)


@app.get("/cron/examples")
def cron_examples():
    """Get common cron expression examples"""
    return {
        "examples": [
            {
                "expression": "0 9 * * *",
                "description": "Every day at 9:00 AM",
                "fields": {
                    "minute": "0",
                    "hour": "9",
                    "day": "*",
                    "month": "*",
                    "day_of_week": "*",
                },
            },
            {
                "expression": "*/15 * * * *",
                "description": "Every 15 minutes",
                "fields": {
                    "minute": "*/15",
                    "hour": "*",
                    "day": "*",
                    "month": "*",
                    "day_of_week": "*",
                },
            },
            {
                "expression": "0 */2 * * *",
                "description": "Every 2 hours",
                "fields": {
                    "minute": "0",
                    "hour": "*/2",
                    "day": "*",
                    "month": "*",
                    "day_of_week": "*",
                },
            },
            {
                "expression": "0 0 * * 0",
                "description": "Every Sunday at midnight",
                "fields": {
                    "minute": "0",
                    "hour": "0",
                    "day": "*",
                    "month": "*",
                    "day_of_week": "0",
                },
            },
            {
                "expression": "0 0 1 * *",
                "description": "First day of every month at midnight",
                "fields": {
                    "minute": "0",
                    "hour": "0",
                    "day": "1",
                    "month": "*",
                    "day_of_week": "*",
                },
            },
            {
                "expression": "30 8 * * 1-5",
                "description": "Every weekday at 8:30 AM",
                "fields": {
                    "minute": "30",
                    "hour": "8",
                    "day": "*",
                    "month": "*",
                    "day_of_week": "1-5",
                },
            },
            {
                "expression": "0 12,18 * * *",
                "description": "Every day at noon and 6 PM",
                "fields": {
                    "minute": "0",
                    "hour": "12,18",
                    "day": "*",
                    "month": "*",
                    "day_of_week": "*",
                },
            },
            {
                "expression": "0 0 * * 1",
                "description": "Every Monday at midnight",
                "fields": {
                    "minute": "0",
                    "hour": "0",
                    "day": "*",
                    "month": "*",
                    "day_of_week": "1",
                },
            },
        ],
        "format": "minute hour day month day_of_week",
        "fields": {
            "minute": "0-59",
            "hour": "0-23",
            "day": "1-31",
            "month": "1-12",
            "day_of_week": "0-6 (0=Monday, 6=Sunday)",
        },
        "special_characters": {
            "*": "any value",
            ",": "list of values (e.g., 1,3,5)",
            "-": "range of values (e.g., 1-5)",
            "/": "step values (e.g., */15 for every 15)",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
