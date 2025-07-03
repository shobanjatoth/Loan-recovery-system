from dataclasses import dataclass
from typing import Any


@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str



@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str


from dataclasses import dataclass

@dataclass
class DataTransformationArtifact:
    transformed_data_path: str
    transformer_object_path: str
    transformed_train_file_path: str



@dataclass
class ModelTrainerArtifact:
    model_path: str
    test_array_path: str
    accuracy: float
    roc_auc: float


@dataclass
class ModelEvaluationArtifact:
    report_file_path: str
    accuracy: float
    roc_auc: float






