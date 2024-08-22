from Mlproject import logger
from Mlproject.pipeline.stage_data_ingestion import DataIngestionTrainingPipeline
from Mlproject.pipeline.stage_data_validation import DataValidationTrainingPipeline
from Mlproject.pipeline.stage_data_transformation import DataTransformationPipeline
from Mlproject.pipeline.stage_model_trainer import ModelTrainingPipeline
from Mlproject.pipeline.stage_model_evaluation import ModelEvaluationPipeline


STAGE_NAME = 'Data Ingestion Stage'

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = 'Data Validation Stage'

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataValidationTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_trasnformation = DataTransformationPipeline()
   data_trasnformation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Training stage"
try:
   logger.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<")
   ModelTrainingPipeline().main()
   logger.info(f">>>>>>>>>>>{STAGE_NAME} completed successfully<<<<<<<<<")
except Exception as e:
   logger.exception(f"An error occurred in {STAGE_NAME}: {str(e)}")
   raise e

STAGE_NAME = "Model Evaluation stage"
try:
   logger.info(f">>>>>>>>>>>Starting {STAGE_NAME}<<<<<<<<<<")
   ModelEvaluationPipeline().main()
   logger.info(f">>>>>>>>>>>{STAGE_NAME} completed successfully<<<<<<<<<")
except Exception as e:
   logger.exception(f"An error occurred in {STAGE_NAME}: {str(e)}")
   raise e

