import os
from datetime import datetime
from dotenv import load_dotenv

# 🔄 Load environment variables
load_dotenv()


import os
from datetime import date

DATABASE_NAME = "stock_project"
COLLECTION_NAME = "stocks"

MONGODB_URL_key = os.getenv("MONGODB_URL")

PIPELINE_NAME: str = "src"
ARTIFACT_DIR: str = "artifact"

FILE_NAME: str = "loan.csv"

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
MODEL_CONFIG_FILE_PATH = os.path.join("config", "model.yaml")

# Data Ingestion Constants
DATA_INGESTION_COLLECTION_NAME: str = COLLECTION_NAME  # corrected
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"

# Data vallidation

DATA_VALIDATION_DIR_NAME = "data_validation"

# Data Transformation Constants
DATA_TRANSFORMATION_DIR = "data_transformation"
TRANSFORMED_DATA_FILE = "transformed_data.csv"
TRANSFORMER_OBJECT_FILE = "transformer.pkl"
TRANSFORMED_TRAIN_FILE = "transformed_train.npy"

# Model Trainer

MODEL_TRAINER_DIR = "model_trainer"
MODEL_FILE_NAME = "risk_classifier.pkl"
TEST_ARRAY_FILE_NAME = "test.npy"         # ✅ (optional but recommended)
REPORT_FILE_NAME = "report.txt"           # If you use text report (not YAML)


# Model Evaluation
# -------------------------------
MODEL_EVALUATION_FILE_NAME = "report.yaml"






