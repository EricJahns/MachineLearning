import torch

class Metrics():
    def __init__(self):
        self.threshold = 0.7

    def pixel_accuracy(self, y_true, y_pred):
        """
        Pixel accuracy metric
        """
        y_pred = torch.where(y_pred > self.threshold, 1, 0)
        y_pred = y_pred.view(-1, )
        y_true = y_true.view(-1, ).float()
        
        eps = 1e-8
        tp = torch.sum(y_true * y_pred)
        tn = torch.sum((1 - y_true) * (1 - y_pred))
        fn = torch.sum(y_true * (1 - y_pred))
        fp = torch.sum((1 - y_true) * y_pred)
        return (tp + tn + eps) / (tp + tn + fn + fp + eps)
    
    def IoU(self, y_true, y_pred):
        """
        Intersection over Union metric
        """
        eps = 0.0001
        intersection = torch.dot(y_pred.view(-1), y_true.view(-1))
        union = torch.sum(y_pred) + torch.sum(y_true) - intersection + eps
        return (intersection.float() + eps) / union.float()
        
    
    def dice_score(self, y_true, y_pred):
        """
        Dice score metric
        """
        eps = 0.0001
        self.inter = torch.dot(y_pred.view(-1), y_true.view(-1))
        self.union = torch.sum(y_pred) + torch.sum(y_true) + eps
        t = (2 * self.inter.float() + eps) / self.union.float()
        return t