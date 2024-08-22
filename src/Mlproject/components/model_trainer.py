import pandas as pd
import os
from Mlproject import logger
from Mlproject.entity.config_entity import ModelTrainerConfig
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        scale = StandardScaler()
        train_x = scale.fit_transform(train_x)
        test_x = scale.fit_transform(test_x)


        lr = LogisticRegression(penalty='elasticnet', 
                        solver='saga', 
                        C=1/self.config.alpha,  
                        l1_ratio=self.config.l1_ratio, 
                        random_state=42)
        lr.fit(train_x, train_y)

        joblib.dump(scale, os.path.join(self.config.root_dir,self.config.scaler_name))
        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))
