# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import typing as tp
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..core.core import Executor, InfoWatcher, Job, R
from ..core.job_environment import JobEnvironment
from ..core.utils import DelayedSubmission, UncompletedJobError
from ..local.local import LocalJobEnvironment


class DebugInfoWatcher(InfoWatcher):
    # pylint: disable=abstract-method
    def register_job(self, job_id: str) -> None:
        pass


class DebugJobEnvironment(JobEnvironment):
    _env = LocalJobEnvironment._env

    def activated(self) -> bool:
        return "SUBMITIT_DEBUG_JOB_ID" in os.environ

    def _requeue(self, countdown: int) -> None:
        pass


# pylint in python 3.6 is confused by generics.
# pylint: disable=no-self-use
class DebugJob(Job[R]):
    watcher = DebugInfoWatcher()

    def __init__(self, submission: DelayedSubmission) -> None:
        job_id = f"DEBUG_{id(submission)}"
        super().__init__(folder="./tmp", job_id=job_id)
        self._submission = submission
        self.cancelled = False

    def submission(self) -> DelayedSubmission:
        return self._submission

    @property
    def num_tasks(self) -> int:
        return 1

    def cancel(self, check: bool = True) -> None:  # pylint: disable=unused-argument
        self.cancelled = True

    def _check_not_cancelled(self) -> None:
        if self.cancelled:
            raise UncompletedJobError(f"Job {self} was cancelled.")

    def results(self) -> List[R]:
        self._check_not_cancelled()
        if self._submission.done():
            return [self._submission._result]
        os.environ["SUBMITIT_DEBUG_JOB_ID"] = self.job_id
        try:
            return [self._submission.result()]
        finally:
            os.environ.pop("SUBMITIT_DEBUG_JOB_ID")

    def exception(self) -> Optional[BaseException]:  # type: ignore
        self._check_not_cancelled()
        try:
            self._submission.result()
            return None
        except Exception as e:
            # Note that we aren't wrapping the error contrary to what is done in
            # other Executors. It makes the stacktrace smaller and debugging easier.
            return e

    def wait(self) -> None:
        # forces execution.
        self.results()

    def done(self, force_check: bool = False) -> bool:  # pylint: disable=unused-argument
        # forces execution, in case the client is waiting on it to become True.
        self.results()
        return self._submission.done()

    @property
    def state(self) -> str:
        if self._submission.done():
            return "DONE"
        if self.cancelled:
            return "CANCELLED"
        return "QUEUED"

    def get_info(self) -> Dict[str, str]:
        return {"STATE": self.state}

    def stdout(self) -> Optional[str]:
        # TODO: should we capture stdout/stderr ? This seems to interfere with PDB.
        return None

    def stderr(self) -> Optional[str]:
        return None


class DebugExecutor(Executor):

    job_class = DebugJob

    def __init__(self, folder: Union[str, Path]):
        super().__init__(folder)

    def _internal_process_submissions(
        self, delayed_submissions: tp.List[DelayedSubmission]
    ) -> tp.List[Job[tp.Any]]:
        return [DebugJob(ds) for ds in delayed_submissions]
