import os
import sys
import pandas as pd
from pandas import DataFrame
from dataclasses import dataclass

from src.exception import USvisaException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.constants import SCHEMA_FILE_PATH
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

class DataValidation:
    def __init__(self, 
                 data_ingestion_artifact: DataIngestionArtifact, 
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)["loan_recovery"]
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def validate_required_columns_exist(self, df: DataFrame) -> bool:
        try:
            missing_columns = []
            expected_columns = (
                self._schema_config["required_columns"] +
                [self._schema_config["target_column"]]
            )
            for column in expected_columns:
                if column not in df.columns:
                    missing_columns.append(column)

            if missing_columns:
                logging.info(f"⚠️ Missing columns: {missing_columns}")
                return False

            return True
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("🚀 Starting data validation")
            df = self.read_data(self.data_ingestion_artifact.feature_store_file_path)

            error_messages = []

            if not self.validate_required_columns_exist(df):
                error_messages.append("❌ Some required columns are missing.")

            validation_status = len(error_messages) == 0
            message = "✅ Data validation successful." if validation_status else " | ".join(error_messages)

            artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=message
            )

            logging.info(f"🧾 Data Validation Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise USvisaException(e, sys)
