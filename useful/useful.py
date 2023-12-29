"""Main module."""

import contextlib
import hashlib
import inspect
import json
import logging
import os
import subprocess
import time
import traceback
import warnings
from sys import platform

import numpy as np
import pandas as pd
from IPython import get_ipython

from useful.connections import (
    CloudLogger,
    FakeCloudLogger,
    Function,
    FunctionData,
    FunctionError,
    Job,
    Statistic,
)
from useful.statistics_check import UsefulStatistics


class Useful:
    """A class for tracking the runtime of a Python job."""

    def __is_running_in_jupyter(self):
        """Detects if is running the library inside a jupyter notebook."""
        return get_ipython() is not None

    def __init__(  # noqa: PLR0913, PLR0915
        self,
        job_name,
        api_key=None,
        workflow_thread=None,
        block_uploads=False,
        workflow_id=None,
        suffix="",
        prefix="",
        ignore_execution_errors=True,
    ):
        """
        Initialize a Useful object which tracks the runtime of a Python job.

        Args:
            job_name (str): The name of the job.
            workflow_thread (list): The workflow thread that this job is a part of.
            block_uploads (bool): Whether or not to block uploads to the cloud.
            workflow_id (str): The workflow ID that this job is a part of.
            api_key (str, optional): The API key to use for uploading to the cloud.
            suffix (str): Suffix string to add to job and workflow thread.
            prefix (str): Prefix string to add to job and workflow thread.
            ignore_execution_errors (bool): Whether to ignore execution errors or not.
        """
        # Initialization
        # core identifiers of a Job
        self.start_timestamp = self.__get_current_micros()
        self.job = prefix + job_name + suffix
        self.workflow_thread = (
            [prefix + thread_name + suffix for thread_name in workflow_thread]
            if workflow_thread
            else None
        )
        self.workflow_id = workflow_id

        # Data upload method
        self.logger = (
            FakeCloudLogger()
            if block_uploads
            else CloudLogger(api_key or os.environ.get("USEFUL_API_KEY"))
        )

        # Error handling method
        self.ignore_execution_errors = ignore_execution_errors

        payload = Job(
            job=self.job,
            job_timestamp=self.start_timestamp,
            workflow_id=self.workflow_id,
            workflow_thread=self.workflow_thread,
        )

        self.logger.upload_job(payload)

        hash_object = hashlib.md5(bytes("jobname: " + job_name, "utf-8"))  # noqa: S324
        code = "__" + hash_object.hexdigest()
        self.seen_in_runtime = set()
        self.thread_signature = [code, "__t" + str(self.start_timestamp)]

        self.function_hash_to_name = {}
        self.logging_order = [logging.ERROR, logging.INFO, logging.DEBUG]

        self.git_url = None
        self.blame = None
        if not self.__is_running_in_jupyter():
            try:
                split_character = "\\" if platform == "win32" else "/"
                call_file = (
                    traceback.format_list(traceback.extract_stack())[0]
                    .split('File "')[1]
                    .split('",')[0]
                    .replace(split_character * 2, split_character)
                )
                cwd = split_character.join(call_file.split(split_character)[:-1])

                git_blame_command = ["git", "blame", call_file]
                git_blame = (
                    subprocess.check_output(git_blame_command, cwd=cwd)  # noqa: S603
                    .decode("utf-8")
                    .replace(",", "")
                )

                def convert_line_to_items(line):
                    name = line.split("(")[1].split("20")[0].strip()
                    line = line.replace(name, "").replace("(", "").split(")")[0]
                    items = line.split()
                    items.insert(1, name)
                    return items

                data = [
                    convert_line_to_items(x)
                    for x in git_blame.replace(
                        "Not Committed Yet", "Not_Committed_Yet"
                    ).split("\n")
                    if len(x) > 0
                ]
                cols = (
                    [
                        "commit",
                        "author",
                        "commit_date",
                        "commit_time",
                        "flag",
                        "line_number",
                    ]
                    if len(data[0]) == 6  # noqa: PLR2004
                    else [
                        "commit",
                        "file_name",
                        "author",
                        "commit_date",
                        "commit_time",
                        "flag",
                        "line_number",
                    ]
                )
                blame_data = pd.DataFrame(data, columns=cols)
                blame_data = blame_data.loc[blame_data["line_number"] > "0"]
                blame_data["author"] = blame_data["author"].apply(
                    lambda x: x.replace(" ", " _")
                )
                blame_data_ref = pd.DataFrame(
                    [
                        [
                            x.replace(" ", "-")
                            for x in blame_data[
                                ["commit_date", "commit_time", "commit", "author"]
                            ].apply(lambda x: " ".join(x), axis=1)
                        ]
                    ],
                    index=["blame"],
                )
                blame_data_ref.columns = [x + 1 for x in blame_data_ref.columns]

                file_location_command = [
                    "git",
                    "rev-parse",
                    "--show-toplevel",
                    call_file,
                ]
                file_location = (
                    subprocess.check_output(
                        file_location_command,  # noqa: S603
                        cwd=cwd,
                    )
                    .decode("utf-8")
                    .replace(",", "")
                    .split("\n")
                )
                file_location = [x.replace("\\", "/") for x in file_location]
                location = file_location[1].replace(file_location[0], "")

                git_branch_command = ["git", "rev-parse", "HEAD"]
                git_branch = (
                    subprocess.check_output(
                        git_branch_command,  # noqa: S603
                        cwd=cwd,
                    )
                    .decode("utf-8")
                    .replace(",", "")
                )

                url_command = ["git", "remote", "get-url", "origin"]
                url = (
                    subprocess.check_output(
                        url_command,  # noqa: S603
                        cwd=cwd,
                    )
                    .decode("utf-8")
                    .replace(".git", "")
                )

                if "@" in url:
                    url = "https://github.com/" + url.split(":")[1]
                self.git_url = f"{url}/blob/{git_branch}{location}".replace("\n", "")
                self.blame = (
                    blame_data_ref.to_json()
                    if isinstance(blame_data_ref, pd.core.frame.DataFrame)
                    else blame_data_ref
                )

            except Exception as e:
                logging.exception(e)  # noqa: TRY401
                warnings.warn(
                    f"Could not authenticate git from {call_file}. Is the there"
                    " a Github repo in the current working directory or the file"
                    " is not committed?",
                    UserWarning,
                    stacklevel=2,
                )
                pass

    def __get_current_micros(self):
        return round(time.time() * 1000000)

    def check(  # noqa: PLR0915
        self, name_tag=None, parallel_runtime=False, check_statistics=True, verbose=0
    ):
        """
        Tracks the runtime and statistics of a function.

        Args:
            name_tag (str, optional): The name of the function.
            parallel_runtime (bool, optional): Whether or not the function is being run
                in parallel.
            check_statistics (bool, optional): Whether or not to check the function
                output statistics.
            verbose (int, optional): The verbosity of the logging.

        Returns:
            The output of the function.
        """
        logging.basicConfig(
            format="%(levelname)s:%(message)s", level=self.logging_order[verbose]
        )

        def decorate(func):  # noqa: PLR0915
            """Decorate the function."""

            def useful_wrapper(*args, **kwargs):  # noqa: PLR0912, PLR0915
                """Wrap the function."""
                # get unique identifier for the call stack
                safe_name_tag = name_tag if name_tag else func.__name__
                hash_object = hashlib.md5(bytes(safe_name_tag, "utf-8"))  # noqa: S324

                element_name = hash_object.hexdigest() + (
                    "_pl" if parallel_runtime else ""
                )
                code = f"__{element_name}__t{str(self.__get_current_micros())}"
                if parallel_runtime:
                    safe_name_tag += "_pl"
                else:
                    indx = 1
                    trial_name_tag = safe_name_tag
                    while trial_name_tag in self.seen_in_runtime:
                        trial_name_tag = f"{safe_name_tag} ({indx})"
                        indx += 1
                    self.seen_in_runtime.add(trial_name_tag)
                    safe_name_tag = trial_name_tag

                self.function_hash_to_name[
                    "__" + hash_object.hexdigest()
                ] = safe_name_tag

                logging.info(" ---------   [WRAPPER]   --------- ")

                # execute code within the callstack
                # TODO: no need to recompute trace, just need better logic to boil it
                # down (this may not be helpful after all)
                trace_back_code = "".join(
                    (
                        "trace = traceback.format_list(traceback.extract_stack());",
                        "indexes = [(i,x) for i,x in enumerate(trace)",
                        "if 'useful_wrapper' in x]; hash_traces = [trace[i+1]",
                        "for i,x in indexes];",
                    )
                )
                exec(  # noqa: S102
                    f"""def quick_trace(): {trace_back_code} return hash_traces"""
                )
                trace = [
                    x.strip().split(" in ")[-1]
                    for x in locals()["quick_trace"]() + [code]
                    if "__" in x
                ]

                logging.info(f"[TIME] start at: {self.__get_current_micros()}")
                function_signature = self.thread_signature[:]
                for item in trace:
                    items = item.split("__t")
                    function_signature.append(items[0])
                    with contextlib.suppress(Exception):
                        function_signature.append("__t" + items[1])
                logging.info(f"[PRE-COMPUTE EVAL] {function_signature}")

                argument_upload = {}
                for elem, val in kwargs.items():
                    argument_upload[f"kwargs_{elem}"] = (
                        UsefulStatistics.statistics_check(val)
                        if not isinstance(val, str)
                        else val
                    )
                for elem, val in enumerate(args):
                    argument_upload[f"args_{elem}"] = (
                        UsefulStatistics.statistics_check(val)
                        if not isinstance(val, str)
                        else val
                    )
                argument_upload = {
                    k: UsefulStatistics.convert_df_to_str(v)
                    if isinstance(v, pd.core.frame.DataFrame)
                    else v
                    for k, v in argument_upload.items()
                }
                # [start] function log here
                lines, code_line_start = inspect.getsourcelines(func)

                execution_string = "".join(
                    (
                        f"def {code}(func, *args, **kwargs): ",
                        f"{trace_back_code} return hash_traces, ",
                        "func(*args, **kwargs)",
                    )
                )
                exec(execution_string)  # noqa: S102  # add in arguments here?
                # error handing
                try:
                    started_at = self.__get_current_micros()
                    tracebacks, out = locals()[code](func, *args, **kwargs)
                    finished_at = self.__get_current_micros()
                    logging.info(f"[TIME] ended at: {finished_at}")
                except Exception as e:
                    # [end] function log here
                    payload = Function(
                        job=self.job,
                        name=safe_name_tag,
                        job_timestamp=self.start_timestamp,
                        started_at=started_at,
                        finished_at=self.__get_current_micros(),
                        error=FunctionError(message=traceback.format_exc()),
                        data=None,
                    )
                    self.logger.upload_function(payload)
                    self.logger.wait()
                    raise e  # noqa: TRY201

                # for printing the traceback, this isn't always necessary
                if verbose >= 1:
                    tracebacks = [
                        x.strip().split(" in ")[-1] for x in tracebacks if "__" in x
                    ]

                    logging.info("[TRUE TRACEBACK LOG] ---")
                    for i, tr in enumerate(tracebacks):
                        if i == 0:
                            logging.info(
                                tr.split("__t")[0] + f" ({tr.split('__t')[1]})"
                            )
                        else:
                            logging.info(
                                " " * i
                                + "|-- "
                                + tr.split("__t")[0]
                                + f" ({tr.split('__t')[1]})"
                            )
                    logging.info("[TRUE TRACEBACK LOG END] ---")

                    logging.info(" --------- [END WRAPPER] --------- ")

                def convert_stat_to_dict(stat):
                    if isinstance(stat, pd.core.frame.DataFrame):
                        logging.info(stat)
                        di = stat.to_dict()
                        return {
                            str(k1): {str(k2): v2 for k2, v2 in v1.items()}
                            for k1, v1 in di.items()
                        }
                    elif isinstance(stat, (int, float, np.number)):
                        return {"number": {"value": stat}}
                    else:
                        return {"string": {"value": str(stat)}}

                try:
                    # checking output of function
                    number_of_return_arguments = 0
                    if (check_statistics) & (type(out) != type(None)):
                        stats = UsefulStatistics.statistics_check(out)
                        if isinstance(stats, list):
                            for i, stat in enumerate(stats):
                                payload = Statistic(
                                    job=self.job,
                                    name=safe_name_tag,
                                    job_timestamp=self.start_timestamp,
                                    return_order=i,
                                    features=convert_stat_to_dict(stat),
                                )
                                self.logger.upload_statistic(payload)
                            number_of_return_arguments = len(stats)
                        else:
                            payload = Statistic(
                                job=self.job,
                                name=safe_name_tag,
                                job_timestamp=self.start_timestamp,
                                return_order=None,
                                features=convert_stat_to_dict(stats),
                            )
                            self.logger.upload_statistic(payload)
                            number_of_return_arguments = 1

                        if verbose:
                            print(f"--- LOGGING STATS {function_signature} ---")
                            print(payload)

                    # [end] function log here
                    payload = Function(
                        job=self.job,
                        name=safe_name_tag,
                        job_timestamp=self.start_timestamp,
                        finished_at=finished_at,
                        started_at=started_at,
                        error=None,
                        data=FunctionData(
                            number_of_return_arguments=number_of_return_arguments,
                            arguments=json.dumps(argument_upload),
                            code="".join(lines),
                            line_start=code_line_start,
                            git_url=self.git_url,
                            blame=self.blame,
                            caller=self.function_hash_to_name[trace[-2].split("__t")[0]]
                            if len(trace) > 1
                            else None,
                        ),
                    )
                    self.logger.upload_function(payload)
                except Exception as e:
                    logging.exception(e)  # noqa: TRY401
                    if not self.ignore_execution_errors:
                        raise e  # noqa: TRY201
                    pass

                self.logger.wait()  # TODO find a way to run this just once
                return out

            return useful_wrapper

        return decorate
