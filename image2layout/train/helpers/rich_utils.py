# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Additional rich ui components"""

import logging
from contextlib import nullcontext
from typing import Iterable, Iterator, List, Optional, TypeVar, Union

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TaskProgressColumn,
    Text,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

T = TypeVar("T")

CONSOLE = Console(width=120)


class ItersPerSecColumn(ProgressColumn):
    """Renders the iterations per second for a progress bar."""

    def __init__(self, suffix="it/s") -> None:
        super().__init__()
        self.suffix = suffix

    def render(self, task) -> Text:
        speed = task.speed
        if speed is None:
            speed_text = "? iters/sec"
        else:
            speed_text = f"{speed:,.1f} iters/sec"
        return Text(
            f"{task.completed}/{task.total} - {speed_text}", style="progress.percentage"
        )


def status(msg: str, spinner: str = "bouncingBall", verbose: bool = False):
    """A context manager that does nothing is verbose is True. Otherwise it hides logs under a message.

    Args:
        msg: The message to log.
        spinner: The spinner to use.
        verbose: If True, print all logs, else hide them.
    """
    if verbose:
        return nullcontext()
    return CONSOLE.status(msg, spinner=spinner)


class ProgressBar(Iterator[T]):
    def __init__(
        self,
        iterable: Iterable[T],
        run_on_local: bool,
        progress_list: List[
            Union[
                BarColumn,
                TaskProgressColumn,
                TextColumn,
                TimeRemainingColumn,
                ItersPerSecColumn,
            ]
        ],
    ):
        self.iterable = iterable
        self.iterator = iter(self.iterable)
        self.total = len(iterable)
        self.current = 0
        self.run_on_local = run_on_local

        if run_on_local:
            self.progress = Progress(*progress_list)
            self.task_id = self.progress.add_task("progress", total=self.total)
            self.progress.start()

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        if self.current < self.total:
            value = next(self.iterator)
            self.current += 1
            if self.run_on_local:
                self.progress.update(self.task_id, completed=self.current)
            return value
        else:
            if self.run_on_local:
                self.progress.stop()
            raise StopIteration


def get_progress(
    iterable: Iterable, description: str, run_on_local: bool = False
) -> ProgressBar:
    """Helper function to return a rich Progress object."""
    progress_list = [
        TextColumn(description),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
    ]
    progress_list += [ItersPerSecColumn()]
    progress_list += [TimeRemainingColumn(elapsed_when_finished=True, compact=True)]
    progress = ProgressBar(iterable, run_on_local, progress_list)  # type: ignore
    return progress
