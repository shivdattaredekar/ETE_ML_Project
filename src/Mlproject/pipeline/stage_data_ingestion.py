from Mlproject import logger
from Mlproject.config.configuration import ConfigurationManager
from Mlproject.components.data_ingestion import DataIngestion 

STAGE_NAME = "Data Ingestion stage"


class DataIngestionTrainingPipeline():
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

if __name__== "__main__":
    try:
        logger.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<")
        DataIngestionTrainingPipeline().main()
        logger.info(f">>>>>>>>>>>{STAGE_NAME} completed successfully<<<<<<<<<")
    except Exception as e:
        logger.exception(f"An error occurred in {STAGE_NAME}: {str(e)}")
        raise e