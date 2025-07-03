# src/entity/config_entity.py

import os
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict
from src.constants import MODEL_TRAINER_DIR, MODEL_FILE_NAME, TEST_ARRAY_FILE_NAME
from src.constants import *
from src.constants import (
    DATA_TRANSFORMATION_DIR,
    TRANSFORMED_DATA_FILE,
    TRANSFORMER_OBJECT_FILE,
    TRANSFORMED_TRAIN_FILE
)

# Global timestamp
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

@dataclass
class DataIngestionConfig:
    training_pipeline_config: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    data_ingestion_dir: str = field(init=False)
    feature_store_file_path: str = field(init=False)
    collection_name: str = field(default=DATA_INGESTION_COLLECTION_NAME)

    def __post_init__(self):
        self.data_ingestion_dir = os.path.join(
            self.training_pipeline_config.artifact_dir,
            DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_FEATURE_STORE_DIR,
            FILE_NAME
        )



@dataclass
class DataValidationConfig:
    training_pipeline_config: TrainingPipelineConfig
    data_validation_dir: str = None

    def __post_init__(self):
        self.data_validation_dir = os.path.join(
            self.training_pipeline_config.artifact_dir,
            DATA_VALIDATION_DIR_NAME
        )


@dataclass
class DataTransformationConfig:
    training_pipeline_config: 'TrainingPipelineConfig'
    data_transformation_dir: str = None
    transformed_data_path: str = None
    transformer_object_path: str = None
    transformed_train_file_path: str = None  # ✅ For .npy

    def __post_init__(self):
        self.data_transformation_dir = os.path.join(
            self.training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR
        )
        self.transformed_data_path = os.path.join(
            self.data_transformation_dir, TRANSFORMED_DATA_FILE
        )
        self.transformer_object_path = os.path.join(
            self.data_transformation_dir, TRANSFORMER_OBJECT_FILE
        )
        self.transformed_train_file_path = os.path.join(
            self.data_transformation_dir, TRANSFORMED_TRAIN_FILE
        )
# === config_entity.py ===



@dataclass
class ModelTrainerConfig:
    training_pipeline_config: 'TrainingPipelineConfig'  # type hint if circular import
    model_trainer_dir: str = None
    model_path: str = None
    test_array_path: str = None
    transformed_train_file_path: str = None

    def __post_init__(self):
        self.model_trainer_dir = os.path.join(
            self.training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR
        )

        self.model_path = os.path.join(self.model_trainer_dir, MODEL_FILE_NAME)

        self.test_array_path = os.path.join(self.model_trainer_dir, TEST_ARRAY_FILE_NAME)

        self.transformed_train_file_path = os.path.join(
            self.training_pipeline_config.artifact_dir, "data_transformation", "transformed_train.npy"
        )




@dataclass
class ModelEvaluationConfig:
    report_file_path: str
