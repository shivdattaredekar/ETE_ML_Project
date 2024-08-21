from Mlproject import logger
from Mlproject.config.configuration import ConfigurationManager
from Mlproject.components.model_evaluation import ModelEvaluation 

STAGE_NAME = "Model Evaluation stage"


class ModelEvaluationPipeline():
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_evaluation_config()
        model_training = ModelEvaluation(config=model_training_config)
        model_training.evaluate()
        
if __name__== "__main__":
    try:
        logger.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<")
        ModelEvaluationPipeline().main()
        logger.info(f">>>>>>>>>>>{STAGE_NAME} completed successfully<<<<<<<<<")
    except Exception as e:
        logger.exception(f"An error occurred in {STAGE_NAME}: {str(e)}")
        raise e