import sys
import pandas as pd
import numpy as np
from typing import Optional
from src.configuration.mongo_db_connection import MongoDBClient
from src.exception import USvisaException
from src.logger import logging

class USvisaData:
    """
    Provides functionality to extract data from MongoDB and return it as a Pandas DataFrame.
    """

    def __init__(self):
        try:
            self.mongo_client = MongoDBClient()
        except Exception as e:
            raise USvisaException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Export the specified MongoDB collection to a Pandas DataFrame.
        """
        try:
            if database_name:
                collection = self.mongo_client.client[database_name][collection_name]
            else:
                collection = self.mongo_client.database[collection_name]

            df = pd.DataFrame(list(collection.find()))
            logging.info(f"📊 Extracted {len(df)} records from collection '{collection_name}'")

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise USvisaException(e, sys)
