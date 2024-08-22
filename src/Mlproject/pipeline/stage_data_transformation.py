from Mlproject import logger
from Mlproject.config.configuration import ConfigurationManager 
from Mlproject.components.data_transformation import DataTransformation
from pathlib import Path


STAGE_NAME = "Data Transformation stage"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):

        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                    status = f.read().replace('\n'," ").replace(': ',':').split(' ')[:-1]
            
            Status = 'True'
            for item in status[:-1]:
                if item:
                    key,value = item.split(':')
                    if value.strip() != 'True':
                        Status = 'False'
                        print(f'Data will not transform as validation failed in {key}')
                    else:
                        pass
            
            if Status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(data_transformation_config)
                data_transformation.TrainTestSplit()
                data_transformation.scaling()
            else :
                raise Exception("Data will not transform as validation failed check running logs for more info")

        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e