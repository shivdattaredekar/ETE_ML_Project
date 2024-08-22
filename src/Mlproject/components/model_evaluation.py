
from Mlproject.entity.config_entity import ModelEvaluationConfig
from Mlproject.utils.common import *


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix
)
import pandas as pd

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metric(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred)
        log_loss_value = log_loss(actual, pred)
        

        return accuracy, precision, recall, f1, roc_auc, log_loss_value

    def evaluate(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        scaler = joblib.load(self.config.scaler_path)
        X_test = test_data.drop([self.config.target_column], axis=1)
        
        y_test = test_data[[self.config.target_column]]
        
        X_test = scaler.transform(X_test)
        y_pred = model.predict(X_test)
        accuracy, precision, recall, f1, roc_auc, log_loss_value = self.eval_metric(y_test,y_pred)

        scores = {'accuracy score':accuracy, 'Precision score':precision, 'Recall score':recall,
                  'F1 score':f1, 'ROC score':roc_auc, 'Log loss':log_loss_value}
        
        save_json(path = Path(self.config.metric_file_name), data = scores)


