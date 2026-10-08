import logging

from castinspect.logging import setup_logging


def test_sets_root_level() -> None:
    setup_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG


def test_accepts_int_level() -> None:
    setup_logging(logging.WARNING)
    assert logging.getLogger().level == logging.WARNING


def test_repeated_setup_keeps_single_handler() -> None:
    setup_logging()
    setup_logging()
    assert len(logging.getLogger().handlers) == 1
