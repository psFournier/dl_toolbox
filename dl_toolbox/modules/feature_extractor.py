import torch.nn as nn
import timm

class FeatureExtractor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        """ Careful: 
        - use_fc_norm is 
            - (global_pool in avg, max, avgmax) if fc_norm param set to None
            - fc_norm param if not None
        - if use_fc_norm set to True, no norm after blocks, but norm after pooling of output tokens
        - if not use_fc_norm, norm after blocks right before pooling, no norm after pooling
        Models are pretrained with fc_norm=False, so we need to set the same and remove the layers after.
        """
        self.encoder = timm.create_model(
            encoder, # Name of the pretrained model
            pretrained=True,
            dynamic_img_size=True, # True deals with image sizes != from pretraining, but cause an issue with create_feature_extractor
            fc_norm=False
        )
        #self.encoder.prune_intermediate_layers(11,prune_norm=True,prune_head=True)
        
    def forward(self, x):
        return self.encoder.forward_features(x)