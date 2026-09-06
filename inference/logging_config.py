import logging
import sys

from pythonjsonlogger.json import JsonFormatter


def setup_logging() -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)

    formatter = JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s "
        "%(request_id)s %(method)s %(path)s %(status_code)s "
        "%(duration_ms)s %(generation_time_seconds)s"
    )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


setup_logging()
