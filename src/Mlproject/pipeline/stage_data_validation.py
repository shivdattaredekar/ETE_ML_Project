from Mlproject import logger
from Mlproject.config.configuration import ConfigurationManager
from Mlproject.components.data_validation import DataValidation 

STAGE_NAME = "Data Validation stage"


class DataValidationTrainingPipeline():
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()
        data_validation.validate_column_types()
        data_validation.validate_missing_values()
if __name__== "__main__":
    try:
        logger.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<")
        DataValidationTrainingPipeline().main()
        logger.info(f">>>>>>>>>>>{STAGE_NAME} completed successfully<<<<<<<<<")
    except Exception as e:
        logger.exception(f"An error occurred in {STAGE_NAME}: {str(e)}")
        raise e