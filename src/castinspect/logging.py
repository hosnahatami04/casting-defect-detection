import logging
import sys

_FORMAT = "%(asctime)s %(levelname)s %(name)s :: %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def setup_logging(level: int | str = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    # Replace handlers so repeated calls stay idempotent.
    root.handlers[:] = [handler]
    root.setLevel(level)
