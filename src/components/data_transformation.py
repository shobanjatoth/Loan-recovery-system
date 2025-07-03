import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans

from src.logger import logging
from src.exception import USvisaException
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact
from src.utils.main_utils import save_object, read_yaml_file
from src.constants import SCHEMA_FILE_PATH


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)["loan_recovery"]
        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            num_features = [col for col, dtype in self.schema_config["column_dtypes"].items() if dtype in ["int", "float"]]
            cat_features = [col for col, dtype in self.schema_config["column_dtypes"].items() if dtype == "str"]

            transformer = ColumnTransformer([
                ("num", StandardScaler(), num_features),
                ("cat", OneHotEncoder(handle_unknown='ignore'), cat_features)
            ])

            return transformer
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("📊 Starting data transformation step")
            df = pd.read_csv(self.data_ingestion_artifact.feature_store_file_path)
            df.drop(columns=self.schema_config["dropped_columns"], inplace=True)

            # 📈 KMeans clustering
            cluster_features = [
                'Age', 'Monthly_Income', 'Loan_Amount', 'Loan_Tenure', 'Interest_Rate',
                'Collateral_Value', 'Outstanding_Loan_Amount', 'Monthly_EMI',
                'Num_Missed_Payments', 'Days_Past_Due'
            ]

            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df[cluster_features])

            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            df['Borrower_Segment'] = kmeans.fit_predict(df_scaled)
            df['Segment_Name'] = df['Borrower_Segment'].map({
                0: 'Moderate Income, High Loan Burden',
                1: 'High Income, Low Default Risk',
                2: 'Moderate Income, Medium Risk',
                3: 'High Loan, Higher Default Risk'
            })

            df['High_Risk_Flag'] = df['Segment_Name'].apply(
                lambda x: 1 if x in ['High Loan, Higher Default Risk', 'Moderate Income, High Loan Burden'] else 0
            )

            # 🔄 Apply transformers
            transformer = self.get_data_transformer_object()
            transformed_array = transformer.fit_transform(df[self.schema_config["required_columns"]])

            os.makedirs(os.path.dirname(self.data_transformation_config.transformer_object_path), exist_ok=True)
            save_object(self.data_transformation_config.transformer_object_path, transformer)

            # 🧪 Final transformed DataFrame
            transformed_df = pd.DataFrame(
                transformed_array.toarray() if hasattr(transformed_array, "toarray") else transformed_array
            )
            transformed_df["High_Risk_Flag"] = df["High_Risk_Flag"].values
            transformed_df["Segment_Name"] = df["Segment_Name"].values

            # Save to .csv
            transformed_df.to_csv(self.data_transformation_config.transformed_data_path, index=False)

            # Save to .npy for training
            train_array = transformed_df.drop(columns=["Segment_Name"]).values
            np.save(self.data_transformation_config.transformed_train_file_path, train_array)

            logging.info("✅ Data transformation complete")

            return DataTransformationArtifact(
                transformed_data_path=self.data_transformation_config.transformed_data_path,
                transformer_object_path=self.data_transformation_config.transformer_object_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path
            )

        except Exception as e:
            raise USvisaException(e, sys)
