#import torch
#import torch.nn.functional as F
#from torch import Tensor
#from torch.nn.modules.loss import _Loss
#
#
#class SumOfLosses(_Loss):
#    """
#    Implementation of Dice loss for image segmentation task.
#    It supports binary, multiclass and multilabel cases
#    """
#
#    def __init__(
#        self,
#        losses,
#        weights,
#    ):
#        super(SumOfLosses, self).__init__()
#        self.losses = losses
#        
#    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
#        """
#
#        :param y_pred: NxCxHxW
#        :param y_true: NxHxW
#        :return: scalar
#        """
#        loss_values = [loss(y_pred, y_true) for loss in self.losses]
#        return [w*l for w,l in zip(self.weights, loss_values)]
