import os
import pandas as pd
from Mlproject import logger
from Mlproject.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config : DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = True

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)
            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    logger.error(f"Column {col} not found in schema.")
                    break  

            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}\n")

            return validation_status
        
        except Exception as e:
            logger.error(f"Error during column validation: {str(e)}")
            raise e

        

    def validate_column_types(self)-> bool:
        try:
            type_status = True
            data = pd.read_csv(self.config.unzip_data_dir)
            
            for col, expected_dtype in self.config.all_schema.items():
                if col in data.columns.to_list():
                    if data[col].dtype != expected_dtype:
                        type_status = False
                        logger.error(f"Column {col} has a different data type: expected {expected_dtype}, found {data[col].dtype}")
                        break  
                    
                else:
                    type_status = True
                    
            with open(self.config.STATUS_FILE, 'a') as f:
                f.write(f"Type status: {type_status}\n")
            return type_status
        except Exception as e:
            raise e
        