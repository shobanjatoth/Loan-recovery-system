# src/components/data_ingestion.py

import os
import sys
import pandas as pd
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import USvisaException
from src.logger import logging
from src.data_access.data_exe import USvisaData


class DataIngestion:
    """
    Handles data extraction from MongoDB and stores it in a local feature store.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
            logging.info(f"📁 Data Ingestion config initialized: {self.data_ingestion_config}")
        except Exception as e:
            raise USvisaException(e, sys)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        Export data from MongoDB to a CSV file in the feature store directory.
        """
        try:
            logging.info("📥 Exporting data from MongoDB to feature store")
            usvisa_data = USvisaData()
            dataframe = usvisa_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )

            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_path), exist_ok=True)
            dataframe.to_csv(feature_store_path, index=False, header=True)

            logging.info(f"✅ Data exported to CSV at: {feature_store_path}")
            return dataframe
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the full ingestion process and returns artifact path.
        """
        try:
            dataframe = self.export_data_into_feature_store()
            return DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            )
        except Exception as e:
            raise USvisaException(e, sys)

