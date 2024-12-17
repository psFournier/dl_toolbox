import torch

class CrossEntropy(torch.nn.CrossEntropyLoss):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.__name__ = 'CE'

    def forward(self, logits, targets):
        return super().forward(logits, targets)
        
    def prob(self, logits):
        return logits.softmax(dim=1)
    
    def pred(self, probs):
        return torch.max(probs, dim=1)