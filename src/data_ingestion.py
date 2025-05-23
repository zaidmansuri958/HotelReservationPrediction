import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml
import shutil


logger=get_logger(__name__)


class DataIngestion:
    def __init__(self,config):
        self.config=config["data_ingestion"]
        self.file_name=self.config["file_name"]
        self.train_test_ratio=self.config["train_ratio"]

        os.makedirs(RAW_DIR,exist_ok=True)
        logger.info(f"Data ingestion started with {self.file_name}")

    def get_raw_data(self):
        try:
            src_path="C:/Users/ZAID/Desktop/MLops/HotelReservationPrediction/data"
            logger.info("Getting the data into the raw file")
            for filename in os.listdir(src_path):
                src_file = os.path.join(src_path, filename)
                if os.path.isfile(src_file):
                    shutil.copy(src_file, RAW_FILE_PATH)
        except Exception as e:
            logger.error(f"Error while copying data {e}")
            raise CustomException("Failed to copy the data ",e)
        
    def split_data(self):
        try:
            logger.info("Start splitting process")
            data=pd.read_csv(RAW_FILE_PATH)
            train_data,test_data=train_test_split(data,train_size=self.train_test_ratio,random_state=42)

            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)

            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")

        except Exception as e:
            logger.error("Error while splitting the data")
            raise CustomException("Failed to split the data")
    
    def run(self):
        try:
            logger.info("data ingestion process started")
            self.get_raw_data()
            self.split_data()

            logger.info("Data ingestion process ended")
        except Exception as e:
            logger.error(f"CustomException : {e}")

if __name__ == "__main__" :
    data_ingestion=DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()           

