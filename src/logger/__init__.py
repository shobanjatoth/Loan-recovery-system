import logging
import os
from datetime import datetime


# Constants
LOG_DIR = "logs"
LOG_FILE = "app.log"

# Ensure log directory exists (even in Docker)
os.makedirs(LOG_DIR, exist_ok=True)

# Final log file path
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Basic configuration
logging.basicConfig(
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    level=logging.INFO
)

# You can now use: from src.logger import logging

