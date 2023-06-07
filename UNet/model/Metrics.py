import numpy as np


class Metrics():
    def __init__(self):
        self.threshold = 0.7

    def pixel_accuracy(self, y_true, y_pred):
        """
        Pixel accuracy metric
        """
        y_pred = np.where(y_pred > self.threshold, 1, 0)
        eps = 1e-8
        tp = np.sum(y_true * y_pred)
        tn = np.sum((1 - y_true) * (1 - y_pred))
        fn = np.sum(y_true * (1 - y_pred))
        fp = np.sum((1 - y_true) * y_pred)
        return (tp + tn + eps) / (tp + tn + fn + fp + eps)
    
    def IoU(self, y_true, y_pred):
        """
        Intersection over Union metric
        """
        y_pred = np.where(y_pred > self.threshold, 1, 0)
        intersection = np.logical_and(y_true, y_pred)
        union = np.logical_or(y_true, y_pred)
        return np.sum(intersection) / np.sum(union)
    
    def dice_score(self, y_true, y_pred):
        """
        Dice score metric
        """
        y_pred = np.where(y_pred > self.threshold, 1, 0)
        intersection = np.logical_and(y_true, y_pred)
        return 2. * np.sum(intersection) / (np.sum(y_true) + np.sum(y_pred))