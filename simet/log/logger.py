"""Structured logging utilities.

Provides:
- `JSONFormatter`: a logging.Formatter that emits JSON lines with ISO-8601 UTC
  timestamps, optional field remapping, exception/stack info, and all custom
  `LogRecord` extras.
- `NonErrorFilter`: a filter that only allows records up to and including INFO
  level (i.e., it filters out WARNING and above).
"""

import datetime as dt
import json
import logging
from typing import override

LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JSONFormatter(logging.Formatter):
    """Emit log records as JSON lines with UTC timestamps.

    The formatter builds a dictionary from the `LogRecord`, adds a UTC
    ISO-8601 timestamp and the rendered `message`, optionally **remaps**
    fields according to `fmt_keys`, and finally injects any custom
    `record` attributes (i.e., extras not in the built-in set).

    By default, the output includes:
      - `"message"`: `record.getMessage()`
      - `"timestamp"`: `record.created` formatted as ISO-8601 in UTC
      - `"exc_info"` and/or `"stack_info"` when present
      - all **custom** attributes found on the record (extras)

    Args:
        fmt_keys (dict[str, str] | None):
            Optional mapping **output_key → source_name** to rename or surface
            specific fields. For each pair:
              * If `source_name` is `"message"` or `"timestamp"`, that value is
                taken from the computed fields and **removed** from the defaults
                (so it won’t also appear under the default key).
              * Otherwise, `getattr(record, source_name)` is used.
            Example:
                `{"level": "levelname", "logger": "name", "time": "timestamp"}`

    Notes:
        - Non-JSON-serializable values are converted with `default=str` in
          `json.dumps`.
        - Timestamps are generated from `record.created` in **UTC**:
          `YYYY-MM-DDTHH:MM:SS.mmmmmm+00:00`.
        - Extras (e.g., provided via `logger.info("..", extra={"run_id": 7})`)
          are included as top-level keys unless they collide with remapped keys.

    Example:
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(JSONFormatter(fmt_keys={"level": "levelname", "logger": "name"}))
        >>> logger = logging.getLogger("app")
        >>> logger.addHandler(handler)
        >>> logger.setLevel(logging.INFO)
        >>> logger.info("hello", extra={"run_id": 123})  # doctest: +SKIP
        {"level": "INFO", "logger": "app", "message": "hello", "timestamp": "...", "run_id": 123}
    """

    def __init__(
        self,
        *,
        fmt_keys: dict[str, str] | None = None,
    ):
        """Initialize the JSON formatter.

        Args:
            fmt_keys: Optional output_key → source_name mapping; see class docstring.
        """
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Format a `LogRecord` as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            str: A single JSON line representing the record.
        """
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict:
        """Build the intermediate dictionary that will be serialized to JSON.

        This method:
          1) Creates the `always_fields` (`message`, `timestamp`, plus `exc_info`
             and `stack_info` when present).
          2) Applies `fmt_keys` remapping (populating requested output keys and
             removing remapped items from `always_fields` to avoid duplication).
          3) Merges remaining `always_fields`.
          4) Injects any custom record attributes not in `LOG_RECORD_BUILTIN_ATTRS`.

        Args:
            record: The log record being formatted.

        Returns:
            dict: A dictionary ready to be passed to `json.dumps`.
        """
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
        }
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: msg_val
            if (msg_val := always_fields.pop(val, None)) is not None
            else getattr(record, val)
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message


class NonErrorFilter(logging.Filter):
    """Allow only log records up to and including INFO level.

    Useful to split output by severity, e.g., one handler for INFO-and-below
    and another handler for WARNING-and-above.

    Returns `True` for levels **<= INFO**, causing the record to pass through
    the handler this filter is attached to; returns `False` otherwise.

    Example:
        >>> info_handler = logging.StreamHandler()
        >>> info_handler.addFilter(NonErrorFilter())  # passes DEBUG/INFO
        >>> err_handler = logging.StreamHandler()
        >>> err_handler.setLevel(logging.WARNING)     # handles WARNING+

        >>> log = logging.getLogger("app")
        >>> log.addHandler(info_handler)
        >>> log.addHandler(err_handler)
        >>> log.info("hello")    # goes to info_handler
        >>> log.warning("oops")  # goes only to err_handler
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        """Return True iff the record's level is <= INFO.

        Args:
            record: The log record to test.

        Returns:
            bool | logging.LogRecord: True to allow the record; False to drop it.
        """
        return record.levelno <= logging.INFO
