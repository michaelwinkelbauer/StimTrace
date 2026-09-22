"""Reusable Qt background workers for non-GUI operations."""
from __future__ import annotations

import json
import ssl
import subprocess
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QThread, Signal

from app_logging import get_logger


LOGGER = get_logger("background")


def is_external_service_error(error: Exception) -> bool:
    """Identify expected network/auth failures without logging a full traceback loop."""
    return isinstance(error, (ConnectionError, TimeoutError, ssl.SSLError)) or (
        error.__class__.__module__.startswith(("google.", "urllib3.", "httplib2"))
    )


class BackgroundFunctionThread(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    progress = Signal(object)

    def __init__(self, function: Callable[[], Any], parent=None) -> None:
        super().__init__(parent)
        self.function = function

    def run(self) -> None:
        try:
            self.succeeded.emit(self.function())
        except Exception as error:
            if is_external_service_error(error):
                LOGGER.warning("Background operation unavailable: %s", error)
            else:
                LOGGER.exception("Background operation failed")
            self.failed.emit(str(error))


class LocalProcessThread(QThread):
    event_received = Signal(dict)

    def __init__(self, program: str, arguments: list[str], parent=None) -> None:
        super().__init__(parent)
        self.program = program
        self.arguments = arguments
        self.process: subprocess.Popen | None = None
        self.stop_requested = False
        self.exit_code = -1
        self.output_tail = ""

    def run(self) -> None:
        output_tail: list[str] = []
        if self.stop_requested:
            self.exit_code = -2
            self.output_tail = "Process start cancelled."
            return
        try:
            creation_flags = (
                subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            )
            self.process = subprocess.Popen(
                [self.program, *self.arguments],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
            )
            if self.stop_requested:
                self.terminate_process()
            assert self.process.stdout is not None
            for raw_line in self.process.stdout:
                line = raw_line.rstrip()
                if not line:
                    continue
                if line.startswith("STIMTRACE_EVENT "):
                    try:
                        self.event_received.emit(json.loads(line.removeprefix("STIMTRACE_EVENT ")))
                    except (TypeError, ValueError):
                        LOGGER.warning("Ignoring malformed local-worker event: %s", line)
                else:
                    output_tail.append(line)
                    del output_tail[:-40]
            self.exit_code = self.process.wait()
        except Exception as error:
            LOGGER.exception("Local worker process failed")
            self.exit_code = -1
            output_tail.append(str(error))
        finally:
            if self.process is not None and self.process.stdout is not None:
                self.process.stdout.close()
            self.output_tail = "\n".join(output_tail[-20:])
            self.process = None

    def terminate_process(self) -> None:
        self.stop_requested = True
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()

    def kill_process(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.kill()
