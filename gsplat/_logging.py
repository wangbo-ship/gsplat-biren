# SPDX-License-Identifier: Apache-2.0

"""Small project-wide logging helpers for gsplat-biren command output."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Iterable, Sequence


_PROJECT_LOGGER = "gsplat_biren"


class _StdoutProxy:
    """Resolve sys.stdout at emit time so pytest capture and redirects work."""

    def write(self, message: str) -> int:
        return sys.stdout.write(message)

    def flush(self) -> None:
        sys.stdout.flush()


class _GsplatFormatter(logging.Formatter):
    """Single-line prefix with aligned continuation lines for tables."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        short_name = record.name
        if short_name.startswith(f"{_PROJECT_LOGGER}."):
            short_name = short_name[len(_PROJECT_LOGGER) + 1 :]
        prefix = f"{timestamp} | {record.levelname:<7} | {short_name} | "
        message = record.getMessage()
        return prefix + message.replace("\n", "\n" + " " * len(prefix))


def _new_stdout_handler() -> logging.Handler:
    handler = logging.StreamHandler(_StdoutProxy())
    handler.setFormatter(_GsplatFormatter())
    return handler


def configure_logging(level: int | str = logging.INFO, *, force: bool = False) -> None:
    """Configure the gsplat-biren logger without touching unrelated loggers."""

    logger = logging.getLogger(_PROJECT_LOGGER)
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
    if not logger.handlers:
        logger.addHandler(_new_stdout_handler())
    logger.setLevel(level)
    logger.propagate = False

    warnings_logger = logging.getLogger("py.warnings")
    if force:
        for handler in list(warnings_logger.handlers):
            warnings_logger.removeHandler(handler)
    if not warnings_logger.handlers:
        warnings_logger.addHandler(_new_stdout_handler())
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.propagate = False
    logging.captureWarnings(True)


def _ensure_logging() -> None:
    logger = logging.getLogger(_PROJECT_LOGGER)
    if not logger.handlers:
        configure_logging()


def get_logger(name: str) -> logging.Logger:
    """Return a child logger in the project logging namespace."""

    safe_name = name.replace("__main__", "cli").replace(".", "_")
    return logging.getLogger(f"{_PROJECT_LOGGER}.{safe_name}")


def log_info(logger: logging.Logger, message: str) -> None:
    _ensure_logging()
    logger.info(message)


def log_warning(logger: logging.Logger, message: str) -> None:
    _ensure_logging()
    logger.warning(message)


def log_error(logger: logging.Logger, message: str) -> None:
    _ensure_logging()
    logger.error(message)


def gsplat_biren_ascii_art() -> str:
    return "\n".join(
        [
            "  ____ ____  ____  _      _  _____     ____ ___ ____  _____ _   _",
            " / ___/ ___||  _ \\| |    / \\|_   _|   | __ )_ _|  _ \\| ____| \\ | |",
            "| |  _\\___ \\| |_) | |   / _ \\ | |_____|  _ \\| || |_) |  _| |  \\| |",
            "| |_| |___) |  __/| |__/ ___ \\| |_____| |_) | ||  _ <| |___| |\\  |",
            " \\____|____/|_|   |_____/_/   \\_\\_|     |____/___|_| \\_\\_____|_| \\_|",
        ]
    )


def _stringify(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "None"
    return str(value)


def format_key_value_table(
    title: str,
    rows: Iterable[tuple[object, object]],
    *,
    key_header: str = "key",
    value_header: str = "value",
) -> str:
    """Format a compact ASCII key/value table for command-line logs."""

    table_rows = [(str(key), _stringify(value)) for key, value in rows]
    rows_with_header = [(key_header, value_header), *table_rows]
    key_width = max(len(row[0]) for row in rows_with_header)
    value_width = max(len(row[1]) for row in rows_with_header)
    total_width = key_width + value_width + 7
    title_line = f"| {title.center(total_width - 4)} |"
    rule = "+" + "-" * (total_width - 2) + "+"
    sep = f"+-{'-' * key_width}-+-{'-' * value_width}-+"
    lines = [
        rule,
        title_line,
        sep,
        f"| {key_header.ljust(key_width)} | {value_header.ljust(value_width)} |",
        sep,
    ]
    for key, value in table_rows:
        lines.append(f"| {key.ljust(key_width)} | {value.ljust(value_width)} |")
    lines.append(sep)
    return "\n".join(lines)


def format_table(
    title: str,
    headers: Sequence[object],
    rows: Sequence[Sequence[object]],
) -> str:
    """Format a general ASCII table for scripts with tabular summaries."""

    string_headers = [str(item) for item in headers]
    string_rows = [[_stringify(item) for item in row] for row in rows]
    widths = [
        max(len(row[index]) for row in [string_headers, *string_rows])
        for index in range(len(string_headers))
    ]
    total_width = sum(widths) + 3 * len(widths) + 1
    rule = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    title_rule = "+" + "-" * (total_width - 2) + "+"
    lines = [
        title_rule,
        f"| {title.center(total_width - 4)} |",
        rule,
        "| "
        + " | ".join(
            string_headers[index].ljust(widths[index])
            for index in range(len(widths))
        )
        + " |",
        rule,
    ]
    for row in string_rows:
        lines.append(
            "| "
            + " | ".join(row[index].ljust(widths[index]) for index in range(len(widths)))
            + " |"
        )
    lines.append(rule)
    return "\n".join(lines)
