import os
from Mlproject import logger
from Mlproject.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataTransformation:
    def __init__(self, config = DataTransformationConfig):
        self.config = config

    def TrainTestSplit(self):
        data = pd.read_csv(self.config.data_path)

        train, test = train_test_split(data)
        train.to_csv(os.path.join(self.config.root_dir,'train.csv'),index = False)
        test.to_csv(os.path.join(self.config.root_dir,'test.csv'),index = False)

        logger.info('Train and Test split done')
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
    
    def scaling(self):
        # Implement scaling logic here
        train = pd.read_csv(os.path.join(self.config.root_dir,'train.csv'))
        test = pd.read_csv(os.path.join(self.config.root_dir,'test.csv'))

        scale = StandardScaler()
        train = scale.fit_transform(train)
        test = scale.fit_transform(test)


