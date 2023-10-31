import torch

class BCE(torch.nn.BCEWithLogitsLoss):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.__name__ = 'binary cross entropy'

    def forward(self, logits, targets):
        return super().forward(logits, targets)
        
    def prob(self, logits):
        return torch.sigmoid(logits)