from __future__ import annotations

import logging
import os


_CONFIGURED = False
_IS_MAIN_PROCESS: bool | None = None


def configure_logging(*, is_main_process: bool | None = None) -> None:
    """Configure concise rank-aware process logging.

    INFO output is emitted only by rank zero. Warnings and errors remain visible
    from every rank so distributed failures are not hidden.
    """

    global _CONFIGURED, _IS_MAIN_PROCESS
    if is_main_process is None:
        is_main_process = _IS_MAIN_PROCESS if _IS_MAIN_PROCESS is not None else int(os.environ.get("RANK", "0")) == 0
    _IS_MAIN_PROCESS = is_main_process

    if not _CONFIGURED:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        _CONFIGURED = True
    logging.getLogger().setLevel(logging.INFO if is_main_process else logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
