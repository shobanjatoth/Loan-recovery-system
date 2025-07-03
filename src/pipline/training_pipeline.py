import os
import sys
import mlflow

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    TrainingPipelineConfig
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact
)

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

from src.logger import logging
from src.exception import USvisaException
from src.constants import MODEL_EVALUATION_FILE_NAME


class TrainPipeline:
    def __init__(self):
        try:
            logging.info("🚀 Initializing TrainPipeline")

            # ✅ MLflow setup (only once)
            mlflow.set_tracking_uri("https://dagshub.com/shobanjatoth/News-dashboard-MLops.mlflow")
            mlflow.set_experiment("LoanRecoveryExperiment")

            # Pipeline configs
            self.training_pipeline_config = TrainingPipelineConfig()

            self.data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            self.data_validation_config = DataValidationConfig(self.training_pipeline_config)
            self.data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
            self.model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)

            model_eval_dir = os.path.join(self.training_pipeline_config.artifact_dir, "model_evaluation")
            report_path = os.path.join(model_eval_dir, MODEL_EVALUATION_FILE_NAME)
            self.model_evaluation_config = ModelEvaluationConfig(report_file_path=report_path)

        except Exception as e:
            raise USvisaException(e, sys)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("📥 Starting data ingestion...")
        ingestion = DataIngestion(self.data_ingestion_config)
        return ingestion.initiate_data_ingestion()

    def start_data_validation(self, ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        logging.info("🔍 Starting data validation...")
        validation = DataValidation(ingestion_artifact, self.data_validation_config)
        return validation.initiate_data_validation()

    def start_data_transformation(self, ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        logging.info("🔄 Starting data transformation...")
        transformation = DataTransformation(ingestion_artifact, self.data_transformation_config)
        return transformation.initiate_data_transformation()

    def start_model_training(self) -> ModelTrainerArtifact:
        logging.info("🏗️ Starting model training...")
        trainer = ModelTrainer(self.model_trainer_config)
        return trainer.train_model()

    def start_model_evaluation(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact
    ) -> ModelEvaluationArtifact:
        logging.info("📊 Starting model evaluation...")
        evaluator = ModelEvaluation(
            model_trainer_artifact=model_trainer_artifact,
            data_transformation_artifact=data_transformation_artifact,
            model_evaluation_config=self.model_evaluation_config
        )
        return evaluator.initiate_model_evaluation()

    def run_pipeline(self):
        try:
            logging.info("🏁 Pipeline execution started")

            # ✅ Start MLflow run for entire pipeline
            with mlflow.start_run(run_name="LoanRecoveryPipeline"):
                ingestion_artifact = self.start_data_ingestion()

                validation_artifact = self.start_data_validation(ingestion_artifact)
                if not validation_artifact.validation_status:
                    raise Exception("❌ Data validation failed. Stopping pipeline.")

                transformation_artifact = self.start_data_transformation(ingestion_artifact)
                logging.info(f"✅ Data transformation completed.")

                self.model_trainer_config.transformed_train_file_path = transformation_artifact.transformed_train_file_path

                model_trainer_artifact = self.start_model_training()
                logging.info(f"✅ Model training completed.")

                evaluation_artifact = self.start_model_evaluation(
                    model_trainer_artifact=model_trainer_artifact,
                    data_transformation_artifact=transformation_artifact
                )
                logging.info(f"📄 Evaluation Report: {evaluation_artifact}")

        except Exception as e:
            raise USvisaException(e, sys)









