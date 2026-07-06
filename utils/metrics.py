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

    def update(self, predicted, gt):

        for p, g in zip(predicted, gt):
            for level in range(self.target_level + 1):
                game = self._GM(p, g, level)
                self.metrics[level].append(game)

    def compute(self):

        avg_game = {}
        for level, error in self.metrics.items():
            avg_game[level] = sum(error) / len(error)

        return avg_game

    def reset(self):
        self.metrics = {level: [] for level in range(self.target_level + 1)}

    def _GM(self, predicted, gt, level):

        if level == 0:
            return float(
                torch.abs(torch.sum(predicted.detach()) - torch.sum(gt.detach()))
            )

        h, w = predicted.shape[-2], predicted.shape[-1]

        # Handle odd dimensions
        pad_h = 1 if h % 2 != 0 else 0
        pad_w = 1 if w % 2 != 0 else 0

        if pad_h > 0 or pad_w > 0:
            predicted = torch.nn.functional.pad(
                predicted, (0, pad_w, 0, pad_h), mode="constant", value=0
            )
            gt = torch.nn.functional.pad(
                gt, (0, pad_w, 0, pad_h), mode="constant", value=0
            )

        mid_h, mid_w = predicted.shape[-2] // 2, predicted.shape[-1] // 2

        slices_pred = [
            predicted[..., :mid_h, :mid_w],
            predicted[..., :mid_h, mid_w:],
            predicted[..., mid_h:, :mid_w],
            predicted[..., mid_h:, mid_w:],
        ]
        slices_gt = [
            gt[..., :mid_h, :mid_w],
            gt[..., :mid_h, mid_w:],
            gt[..., mid_h:, :mid_w],
            gt[..., mid_h:, mid_w:],
        ]

        total_error = 0.0

        for pred_slice, gt_slice in zip(slices_pred, slices_gt):
            total_error += self._GM(pred_slice, gt_slice, level - 1)

        return total_error
