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


class GameMetrics:
    """
    Calulcate GAME Metric (TRANCOS dataset)
    target_level: int = 3 -> number of game to calculate (numbering starts from 0)
    returns dictionary of each level and average game error
    """

    def __init__(self, target_level=3):
        self.target_level = target_level
        self.metrics = {level: [] for level in range(self.target_level + 1)}

    def update(self, pred_dens, gt_dens):

        for level in range(self.target_level + 1):
            game = self._GM(pred_dens, gt_dens, level)
            self.metrics[level].append(game)

    def compute(self):

        avg_game = {}
        for level, error in self.metrics.items():
            avg_game[level] = sum(error) / len(error)

        return avg_game

    def reset(self):
        self.metrics = {level: [] for level in range(self.target_level + 1)}

    def _GM(self, pred_dens, gt_dens, level):

        if level == 0:
            return float(torch.abs(torch.sum(pred_dens) - torch.sum(gt_dens)))

        h, w = pred_dens.shape

        # Handle odd dimensions
        pad_h = 1 if h % 2 != 0 else 0
        pad_w = 1 if w % 2 != 0 else 0

        if pad_h > 0 or pad_w > 0:
            pred_dens = torch.nn.functional.pad(
                pred_dens, (0, pad_w, 0, pad_h), mode="constant", value=0
            )
            gt_dens = torch.nn.functional.pad(
                gt_dens, (0, pad_w, 0, pad_h), mode="constant", value=0
            )

        mid_h, mid_w = pred_dens.shape[0] // 2, pred_dens.shape[1] // 2

        slices_pred = [
            pred_dens[:mid_h, :mid_w],
            pred_dens[:mid_h, mid_w:],
            pred_dens[mid_h:, :mid_w],
            pred_dens[mid_h:, mid_w:],
        ]
        slices_gt = [
            gt_dens[:mid_h, :mid_w],
            gt_dens[:mid_h, mid_w:],
            gt_dens[mid_h:, :mid_w],
            gt_dens[mid_h:, mid_w:],
        ]

        total_error = 0.0

        for p_slice, g_slice in zip(slices_pred, slices_gt):
            total_error += self.GM(p_slice, g_slice, level - 1)

        return total_error
