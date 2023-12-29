"""Contains tools to connect to cloud services."""

import asyncio
import concurrent
import logging
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import nest_asyncio
import requests
from pydantic import BaseModel


class Job(BaseModel):
    """A representation of a single job."""

    job: str
    job_timestamp: int
    workflow_id: str | None
    workflow_thread: list[str] | None


class FunctionError(BaseModel):
    """A representation of a function error."""

    message: str


class FunctionData(BaseModel):
    """A representation of a function data."""

    arguments: str
    number_of_return_arguments: int | None
    code: str | None
    line_start: int | None
    git_url: str | None
    blame: str | None
    caller: str | None


class Function(BaseModel):
    """A representation of a function."""

    name: str
    job: str
    job_timestamp: int
    finished_at: int
    started_at: int
    error: FunctionError | None
    data: FunctionData | None


class Statistic(BaseModel):
    """A representation of a statistic."""

    job: str
    job_timestamp: int
    name: str
    return_order: int | None
    features: dict[str, dict[str, int | float | str]]


# Enable multithreading inside multithreading like IPython kernel and parallel tasks
nest_asyncio.apply()


class CloudLogger:
    """A class to log data to the cloud."""

    API_ENDPOINT = "https://api.usefulmachines.dev/prod/upload"
    MAX_CONCURRENCY = 10

    def __init__(self, api_key):
        """Set the API and synchronization variables."""
        if api_key is None:
            raise RuntimeError(
                "Authentication needed. Generate a key at: "
                "https://app.usefulmachines.dev/"
            )

        self.api_key = api_key
        self.tasks = []

    def upload_task(self, endpoint, data):
        """Uploads data to endpoints synchronously."""
        requests.post(
            f"{self.API_ENDPOINT}{endpoint}",
            data=data,
            headers={"x-api-key": self.api_key},
            timeout=3,
        )

    async def __wait_async(self):
        """Wait for all tasks to finish asynchronously."""
        with ProcessPoolExecutor(max_workers=self.MAX_CONCURRENCY) as executor:
            try:
                futures = [executor.submit(self.upload_task, *x) for x in self.tasks]
                [concurrent.futures.as_completed(x) for x in futures]
                self.tasks = []

            except BrokenProcessPool:
                # If process pool breaks, reset request handling queue
                logging.warning("Failed to execute requests")
                self.tasks = []
                pass

    def wait(self):
        """Wait for all tasks to finish."""
        asyncio.run(self.__wait_async())

    def upload_job(self, data: Job):
        """Upload a job to the cloud."""
        logging.info(f"[UPLOAD JOB CALLED]: {data}")
        self.tasks.append(("/job", data.model_dump_json(exclude_none=True)))

    def upload_function(self, data: Function):
        """Upload a function to the cloud."""
        logging.info(f"[UPLOAD FUNC CALLED]: {data}")
        self.tasks.append(("/function", data.model_dump_json(exclude_none=True)))

    def upload_statistic(self, data: Statistic):
        """Upload a statistic to the cloud."""
        logging.info(f"[UPLOAD STAT CALLED]: {data}")
        self.tasks.append(("/statistic", data.model_dump_json(exclude_none=True)))


class FakeCloudLogger:
    """A fake cloud logger that does nothing but uploads data to a list attribute."""

    def __init__(self):
        """Initialize the FakeCloudLogger."""
        self.all_data = []

    def wait(self):
        """Simulate a wait."""
        pass

    def upload_job(self, data):
        """Simulate an upload job."""
        self.all_data.append(data.model_dump_json(exclude_none=True))

    def upload_function(self, data):
        """Simulate an upload function."""
        self.all_data.append(data.model_dump_json(exclude_none=True))

    def upload_statistic(self, data):
        """Simulate an upload statistic."""
        self.all_data.append(data.model_dump_json(exclude_none=True))
