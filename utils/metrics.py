import torch 
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class RegressionMetrics:
    """
    Calulcate Regression Metrics : MAE, MSE and R^2 Error 
    """
    def __init__(self):
        self.predictions = []
        self.truths = []

    def update(self, prediction: torch.Tensor, truth: torch.Tensor | int):

        p = prediction.detach().cpu().squeeze().numpy().tolist()
        
        t = []
        if isinstance(truth, torch.Tensor):
            t = truth.detach().cpu().squeeze().numpy().tolist()
        elif isinstance(truth, int): 
            t = list(t)
        
        self.predictions.extend(p)
        self.truths.extend(t)

    def compute(self):

        if not self.predictions:
            return 0.0, 0.0, 0.0
            
        p = np.array(self.predictions)
        t = np.array(self.truths)
        
        mae = mean_absolute_error(t, p)
        mse = mean_squared_error(t, p)
        r2 = r2_score(t, p)
        
        return mae, mse, r2
        
    def reset(self):

        self.predictions = []
        self.truths = []
