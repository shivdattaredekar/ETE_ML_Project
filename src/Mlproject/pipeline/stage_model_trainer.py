from Mlproject import logger
from Mlproject.config.configuration import ConfigurationManager
from Mlproject.components.model_trainer import ModelTrainer 

STAGE_NAME = "Model trainer stage"


class ModelTrainingPipeline():
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_trainer_config()
        model_training = ModelTrainer(config=model_training_config)
        model_training.train()
        
if __name__== "__main__":
    try:
        logger.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<")
        ModelTrainingPipeline().main()
        logger.info(f">>>>>>>>>>>{STAGE_NAME} completed successfully<<<<<<<<<")
    except Exception as e:
        logger.exception(f"An error occurred in {STAGE_NAME}: {str(e)}")
        raise e