import logging
import os
from logging import Logger

_LOGGER: Logger = logging.getLogger("csv_rag")

def configure_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # Uvicorn access logs are noisy; tune as needed
    logging.getLogger("uvicorn.access").setLevel(os.getenv("UVICORN_ACCESS_LOG_LEVEL", "WARNING"))

@property
def logger() -> Logger:
    return _LOGGER
